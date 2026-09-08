"""
ORena SAVE FOCUS challenge — PROCEDURE track.

Cross-model candidate: W64 supplies the broad full-procedure pointer for
single-timestamp temporal questions; NW2 performs both localized refinement
passes and answers every ordinary question. This exact routing improved the
full offline Procedure score from NW2's 58.75% to 59.31%.

Both checkpoints are packaged in the same already-merged form used by offline
evaluation. To stay inside a 24 GB validation GPU, inference is deliberately
two-phase: run every W64 stage-0 pointer, unload W64, then load NW2 for all
remaining work. Only one 4B model is resident at a time.

Near-identical to the SEGMENT container — same model, prompts, infer_format,
absolute-time overlay, and output cleaning. This track's clips are much longer
(median ~53 min), so we sample NUM_FRAMES uniformly across the whole clip.

Inputs (mounted read-only at /input):
  request.json            LIST of focus.Request — one per question
  plain/<qID>.mp4         the video clip (trimmed to [start_time, end_time])
  overlayed/<qID>.mp4     clip-relative clock — NOT used (we re-draw absolute time)
  FO_definitions.json     FO class defs (we use our bundled copy)
  batch.json              convenience index

Output: /output/answer.json — LIST of focus.Response.

See the SEGMENT container docstring for the full rationale (answer_format inference,
bundled system prompt, absolute-time overlay reproducing training, per-question
try/except). The only substantive change here is NUM_FRAMES.
"""

import gc
import logging
import re
import sys
import time
from pathlib import Path

import torch
from focus import Request, Response

RESOURCES_PATH = Path(__file__).parent / "resources"
sys.path.insert(0, str(RESOURCES_PATH))

import prompts  # noqa: E402
from engines import QwenVLEngine  # noqa: E402  (imports torch before decord)
from infer_format import infer_format  # noqa: E402
from overlay_decision import want_overlay  # noqa: E402
from reappearance_presence import route_reappearance_presence  # noqa: E402
from robust_io import SetupFailure, answer_every_question, load_requests_tolerant  # noqa: E402
from video_sampler import sample_clip  # noqa: E402

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
VIDEO_DIR = INPUT_PATH / "plain"
W64_MODEL_PATH = str(RESOURCES_PATH / "w64")
NW2_MODEL_PATH = str(RESOURCES_PATH / "nw2")

NUM_FRAMES = 64           # matches the deployed adapter's training budget (see resources/base/merge_provenance.json)
MAX_LONG_SIDE = 768
MAX_NEW_TOKENS = 64
REAPPEARANCE_NEEDLE_CHUNKS = 4

# ── temporal cascade (single-timestamp time questions only) ──────────────────
# 3-pass HYBRID zoom (gate CH1, 2026-08-05): dense 128f pointer passes, cheap 64f
# narrow final. Full clip @128f -> ±600s @128f around the model's own answer ->
# ±50s @64f. Measured on all 746 single-ts procedure test rows (gold-free):
# 1-pass 0.0684 -> old 3-pass 0.2882 -> hybrid 0.3512 time accuracy
# (+0.0536 vs same-stack control, 80W/40L, p=0.00033; reproduces on the shipping
# spec weights: +0.0657, 93W/44L, p=3.4e-05 — gate CH1-T). Latency p99 27.74s,
# max 28.55s, zero rows over the 30s budget. Density buys the POINTER, narrowing
# buys the FINAL; 128f on the final pass adds latency but no accuracy.
CASCADE_WIDTHS = (1200.0, 100.0)     # full window widths of passes 2 and 3
CASCADE_NUM_FRAMES = (128, 128, 64)  # per stage: full-clip, W=1200, W=100
CASCADE_MAX_NEW_TOKENS = 16          # timestamps only
# The challenge enforces a pooled job allowance, not a hard per-row 30-second
# timeout. The measured cross cascade uses 69.6% of that pool, so every valid
# pointer receives both refinement passes.
_TS_RE = re.compile(r"(\d+):(\d{2}):(\d{2})")


def _ts_seconds(text: str) -> float | None:
    m = _TS_RE.search(text or "")
    if not m:
        return None
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))


def _is_single_time(question: str, fmt: str) -> bool:
    # Multi-event "time points" questions have 1..N golds; zooming on one answer
    # would drop the others. Cascade only the single-timestamp templates.
    return fmt == "time" and "time points" not in (question or "").lower()


def _use_reappearance_chunks(object_name: str) -> bool:
    """Needles alone showed a clean benefit from denser post-anchor coverage."""
    return re.search(r"\bneedle\b", object_name or "", flags=re.IGNORECASE) is not None


def _split_window(
    window: tuple[float, float],
    chunks: int,
) -> list[tuple[float, float]]:
    lo, hi = window
    if chunks <= 1 or hi <= lo:
        return [window]
    width = (hi - lo) / chunks
    return [
        (lo + index * width, hi if index == chunks - 1 else lo + (index + 1) * width)
        for index in range(chunks)
    ]


def _translated_window(answer: str, width: float, req: Request) -> tuple[float, float] | None:
    """Turn an absolute timestamp answer into a fixed-width clip-relative window."""
    clip_dur = float(req.end_time) - float(req.start_time)
    pointer_abs = _ts_seconds(answer)
    if pointer_abs is None:
        return None
    pointer_rel = pointer_abs - float(req.start_time)
    width = min(width, clip_dur)
    lo = pointer_rel - width / 2.0
    if lo < 0.0:
        lo = 0.0
    elif lo + width > clip_dur:
        lo = clip_dur - width
    return lo, lo + width


def answer_time_stage0(engine, clip_path, req, question, system) -> tuple[str, tuple[float, float] | None]:
    """Run W64's broad full-procedure pointer pass."""
    frames = sample_clip(
        clip_path, req.start_time, CASCADE_NUM_FRAMES[0],
        True, MAX_LONG_SIDE, window=None,
    )
    raw = engine.generate(
        frames, question, system, max_new_tokens=CASCADE_MAX_NEW_TOKENS,
    )
    answer = prompts.clean_answer(raw, "time")
    return answer, _translated_window(answer, CASCADE_WIDTHS[0], req)


def answer_time_refine(
    engine, clip_path, req, question, system,
    initial_answer: str, initial_window: tuple[float, float] | None,
) -> str:
    """Run NW2's localized 128-frame and 64-frame refinement passes."""
    answer = initial_answer
    window = initial_window
    if window is None:
        return answer

    for stage in range(1, len(CASCADE_NUM_FRAMES)):
        frames = sample_clip(
            clip_path, req.start_time, CASCADE_NUM_FRAMES[stage],
            True, MAX_LONG_SIDE, window=window,
        )
        raw = engine.generate(frames, question, system,
                              max_new_tokens=CASCADE_MAX_NEW_TOKENS)
        answer = prompts.clean_answer(raw, "time")
        if stage == len(CASCADE_NUM_FRAMES) - 1:
            break
        window = _translated_window(answer, CASCADE_WIDTHS[stage], req)
        if window is None:
            break
    return answer


def _load_engine(model_path: str, label: str, system: str) -> QwenVLEngine:
    """Load and warm one merged checkpoint."""
    from PIL import Image

    t0 = time.monotonic()
    log.info("--- Loading merged %s model (%s) ---", label, model_path)
    engine = QwenVLEngine(model_path, device="cuda")
    engine.load()
    log.info("%s loaded in %.1fs. Warm-up…", label, time.monotonic() - t0)
    warm = engine.generate(
        [Image.new("RGB", (64, 64))],
        "Is a foreign object visible?",
        system,
        max_new_tokens=4,
    )
    log.info("%s warm-up ok -> %r (setup %.1fs)", label, warm, time.monotonic() - t0)
    return engine


def _release_engine(engine: QwenVLEngine, label: str) -> None:
    """Drop all model references and return cached CUDA blocks to the driver."""
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    engine.model = None
    engine.processor = None
    gc.collect()
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass
    log.info(
        "%s released: CUDA allocated %.2f GB, reserved %.2f GB",
        label,
        torch.cuda.memory_allocated() / 2**30,
        torch.cuda.memory_reserved() / 2**30,
    )


def clip_path_for(req: Request) -> Path:
    return VIDEO_DIR / f"{req.qID}.mp4"


def run() -> int:
    t_start = time.monotonic()
    log.info("=== ORena SAVE FOCUS — PROCEDURE inference start ===")

    # Every answer collected so far. `answer_every_question` reads this list on exit
    # and writes one Response per qID in request.json no matter what happens inside
    # the block — a case that produces no answer.json is forfeited outright, which is
    # strictly worse than one that answers badly.
    responses: list[Response] = []
    with answer_every_question(INPUT_PATH / "request.json", OUTPUT_PATH, responses):
        requests = load_requests_tolerant(INPUT_PATH / "request.json")
        if not requests:
            log.error("request.json contains no usable requests")
            return 1
        log.info("Batch of %d question(s)", len(requests))

        system = prompts.system_prompt(include_fo_defs=True, include_knowledge=False)
        log.info("System prompt: %d chars (expect 3276, 10 FO classes)", len(system))
        if len(system) < 500:
            log.error("System prompt too short — bundled FO_definitions.txt missing?")
            return 1

        # Format once before either model phase. Keeping this metadata on CPU
        # lets us execute all W64 pointers together without changing output order.
        prepared: list[dict] = []
        for req in requests:
            try:
                fmt = infer_format(req.question)
                reappearance = route_reappearance_presence(
                    req.question, fmt, req.start_time, req.end_time,
                )
                if reappearance is not None:
                    ov = False
                    question = reappearance.question
                else:
                    ov = want_overlay(req.question, fmt)
                    question = prompts.build_question(req.question, fmt)
                prepared.append({
                    "req": req,
                    "fmt": fmt,
                    "ov": ov,
                    "question": question,
                    "temporal": _is_single_time(req.question, fmt),
                    "reappearance": reappearance,
                    "reappearance_chunks": (
                        reappearance is not None
                        and _use_reappearance_chunks(reappearance.object_name)
                    ),
                    "prepare_error": None,
                })
            except Exception as exc:
                log.exception("qID=%s question preparation failed", req.qID)
                prepared.append({
                    "req": req,
                    "fmt": "?",
                    "ov": False,
                    "question": req.question,
                    "temporal": False,
                    "reappearance": None,
                    "reappearance_chunks": False,
                    "prepare_error": repr(exc),
                })

        temporal_indices = [
            idx for idx, item in enumerate(prepared)
            if item["prepare_error"] is None and item["temporal"]
        ]
        log.info(
            "Routing: %d temporal W64->NW2 cascade, %d direct NW2",
            len(temporal_indices), len(prepared) - len(temporal_indices),
        )

        # Phase 1: W64 only. Store tiny CPU states (answer/window/latency), then
        # fully unload the model before NW2 is instantiated.
        stage0: dict[int, dict] = {}
        if temporal_indices:
            try:
                w64_engine = _load_engine(W64_MODEL_PATH, "W64", system)
            except Exception as exc:
                raise SetupFailure(f"W64 model load / warm-up failed: {exc!r}") from exc

            for idx in temporal_indices:
                item = prepared[idx]
                req = item["req"]
                t0 = time.monotonic()
                try:
                    answer, window = answer_time_stage0(
                        w64_engine, clip_path_for(req), req, item["question"], system,
                    )
                    stage0[idx] = {
                        "answer": answer,
                        "window": window,
                        "latency": time.monotonic() - t0,
                        "failed": False,
                    }
                    log.info(
                        "[W64 %d/%d] qID=%s -> %r window=%s (%.2fs)",
                        temporal_indices.index(idx) + 1,
                        len(temporal_indices),
                        req.qID,
                        answer,
                        window,
                        stage0[idx]["latency"],
                    )
                except Exception:
                    latency = time.monotonic() - t0
                    log.exception("[W64] qID=%s stage-0 failed", req.qID)
                    stage0[idx] = {
                        "answer": "",
                        "window": None,
                        "latency": latency,
                        "failed": True,
                    }

            _release_engine(w64_engine, "W64")
            del w64_engine

        # Phase 2: NW2 answers ordinary rows and refines every successful W64
        # pointer. A broken model environment should fail loudly rather than
        # produce an apparently successful all-empty answer file.
        try:
            nw2_engine = _load_engine(NW2_MODEL_PATH, "NW2", system)
        except Exception as exc:
            raise SetupFailure(f"NW2 model load / warm-up failed: {exc!r}") from exc

        n_failed = 0
        t_batch = time.monotonic()
        for idx, item in enumerate(prepared):
            i = idx + 1
            req = item["req"]
            t0 = time.monotonic()
            stage0_latency = 0.0
            try:
                if item["prepare_error"] is not None:
                    raise RuntimeError(f"question preparation failed: {item['prepare_error']}")
                if item["temporal"]:
                    initial = stage0[idx]
                    stage0_latency = float(initial["latency"])
                    if initial["failed"]:
                        raise RuntimeError("W64 stage-0 failed")
                    answer = answer_time_refine(
                        nw2_engine,
                        clip_path_for(req),
                        req,
                        item["question"],
                        system,
                        initial["answer"],
                        initial["window"],
                    )
                else:
                    reappearance = item["reappearance"]
                    if item["reappearance_chunks"]:
                        subanswers = []
                        for window in _split_window(
                            reappearance.window, REAPPEARANCE_NEEDLE_CHUNKS,
                        ):
                            frames = sample_clip(
                                clip_path_for(req), req.start_time, NUM_FRAMES,
                                False, MAX_LONG_SIDE, window=window,
                            )
                            raw = nw2_engine.generate(
                                frames, item["question"], system,
                                max_new_tokens=MAX_NEW_TOKENS,
                            )
                            subanswer = prompts.clean_answer(raw, "binary")
                            subanswers.append(subanswer)
                            if subanswer == "yes":
                                break
                        answer = (
                            "yes"
                            if "yes" in subanswers
                            else "no"
                            if subanswers and all(value == "no" for value in subanswers)
                            else ""
                        )
                    else:
                        frames = sample_clip(
                            clip_path_for(req), req.start_time, NUM_FRAMES,
                            item["ov"], MAX_LONG_SIDE,
                            window=(
                                reappearance.window
                                if reappearance is not None
                                else None
                            ),
                        )
                        raw = nw2_engine.generate(
                            frames, item["question"], system,
                            max_new_tokens=MAX_NEW_TOKENS,
                        )
                        answer = prompts.clean_answer(raw, item["fmt"])
            except Exception:
                n_failed += 1
                log.exception("[%d/%d] qID=%s failed; empty answer", i, len(requests), req.qID)
                answer = ""
            latency = stage0_latency + (time.monotonic() - t0)
            responses.append(Response(qID=req.qID, content=answer, latency=latency))
            log.info(
                "[%d/%d] qID=%s fmt=%s ov=%s route=%s -> %r (%.2fs)",
                i, len(requests), req.qID, item["fmt"], item["ov"],
                (
                    "W64->NW2"
                    if item["temporal"]
                    else "reappearance-presence-needle-4x64"
                    if item["reappearance_chunks"]
                    else "reappearance-presence"
                    if item["reappearance"] is not None
                    else "NW2"
                ),
                answer, latency,
            )

        log.info("Inference: %d answered (%d failed) in %.1fs",
                 len(responses), n_failed, time.monotonic() - t_batch)
        _release_engine(nw2_engine, "NW2")
        del nw2_engine

    log.info("Total %.1fs", time.monotonic() - t_start)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
