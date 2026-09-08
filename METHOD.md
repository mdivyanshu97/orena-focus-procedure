# DISCOVR-PROCEDURE method

## Overview

DISCOVR-PROCEDURE answers foreign-object questions over long surgical
procedures. Long videos make uniform sampling sparse, so the method separates
broad temporal localization from localized answering:

- **W64** acts as the broad full-procedure timestamp pointer.
- **NW2** performs temporal refinement and answers all non-cascade questions.

Both are merged Qwen3-VL-4B checkpoints. They are loaded sequentially so that
at most one model is resident on the GPU.

## Inputs and output

For each Grand Challenge batch, the container receives:

- `/input/request.json`: one request object per question;
- `/input/plain/<qID>.mp4`: the corresponding procedure clip;
- `/input/overlayed/<qID>.mp4`: a platform-provided relative-time variant;
- `/input/FO_definitions.json` and `/input/batch.json`.

The runtime uses the plain video and draws an absolute-time overlay when
required. It writes an ordered response list to `/output/answer.json`.

## Models

Both checkpoints are bfloat16 merges of rank-16, alpha-32 LoRA adapters into
`Qwen/Qwen3-VL-4B-Instruct`.

| Model | Training export | Serving role |
|---|---|---|
| W64 | `alltracks_train_v2_64f.jsonl` | Broad full-procedure pointer for single-timestamp questions |
| NW2 | `capped_v1_64f.jsonl` | Ordinary answers, post-anchor presence queries, and localized temporal refinement |

The two models share the same prompt construction, answer-format inference,
absolute-time overlay logic, and deterministic output cleanup.

## Batch execution and memory

Question metadata is prepared on CPU before either model phase. The container
then executes:

1. Identify all single-timestamp time questions.
2. Load W64 and run their broad stage-0 pointer passes.
3. Store only the small CPU result for each row: timestamp answer, refinement
   window, latency, and failure state.
4. Release W64, run garbage collection, and clear CUDA caches.
5. Load NW2 once.
6. Refine successful temporal pointers and answer every direct NW2 route.
7. Release NW2 and atomically write the batch output.

If a batch has no single-timestamp questions, W64 is never loaded.

## Routing

| Question route | Model and evidence |
|---|---|
| Single-timestamp time, stage 0 | W64 with 128 frames across the full procedure |
| Single-timestamp time, stage 1 | NW2 with 128 frames in a 1200-second window centered on the W64 timestamp |
| Single-timestamp time, stage 2 | NW2 with 64 frames in a 100-second window centered on the stage-1 timestamp |
| Exact binary reappearance template | NW2 with 64 frames sampled only after the timestamp named in the question |
| Needle reappearance template | NW2 over four chronological post-anchor chunks, 64 frames per chunk, stopping early on “yes” |
| Ordinary question | NW2 with 64 frames sampled across the complete clip |

Window widths are clamped to the available clip. Timestamp predictions are
absolute; they are translated to clip-relative sampling windows before video
decoding.

## Reappearance-presence rewrite

Questions matching the narrow template “Does the object, last visible just
before `<timestamp>`, re-appear later in the video?” are converted to a direct
visual presence question over only the post-timestamp video:

> Looking only at these chronological frames, is the named object visible in
> any frame?

The answer is constrained to `yes` or `no`. Needle questions use four
post-anchor chunks because denser chronological coverage performed better for
that object family. A `yes` in any chunk produces `yes`; all completed `no`
answers produce `no`.

## Temporal cascade

The timestamp route progressively increases evidence density:

1. **Full procedure:** 128 frames establish a broad pointer.
2. **1200-second window:** 128 NW2 frames refine the approximate location.
3. **100-second window:** 64 NW2 frames produce the final timestamp.

Later stages stop if the preceding answer does not contain a valid timestamp.
The cascade is restricted to single-timestamp questions so it cannot collapse a
multi-event timestamp list to one event.

## Prompting and output

The system prompt contains the bundled foreign-object definitions. Each
question receives an inferred answer-format instruction, and generated text is
cleaned into the required binary, number, class, timestamp, or free-text form.

Batch-level hardening:

- tolerates additional request fields;
- isolates per-question failures;
- preserves request order;
- emits one response per recoverable `qID`;
- writes `answer.json` atomically;
- raises model setup failures instead of silently returning an all-empty batch.

## Reproducibility

The repository excludes challenge videos and the two 8.9 GB checkpoints. Run:

```bash
python scripts/download_weights.py
python scripts/verify_release.py
./do_build.sh
```

The downloader pins the model repository revision. The verifier checks the
selected source files and both merged checkpoints against their recorded
SHA-256 values.
