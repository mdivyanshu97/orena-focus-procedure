# Training provenance

Both released checkpoints are bfloat16 merges of LoRA adapters into
`Qwen/Qwen3-VL-4B-Instruct`.

## W64

- LoRA rank / alpha: `16 / 32`;
- primary SFT export: `alltracks_train_v2_64f.jsonl`;
- warm start: SSG auxiliary adapter;
- merge method: `peft.merge_and_unload`;
- serving role: broad full-procedure temporal pointer.

## NW2

- LoRA rank / alpha: `16 / 32`;
- primary SFT export: `capped_v1_64f.jsonl`;
- warm start: SSG auxiliary adapter;
- merge method: `peft.merge_and_unload`;
- serving role: ordinary answers and localized temporal refinement.

The original merge records are published beside each model on Hugging Face.
Raw challenge videos and challenge-provided annotations are not redistributed.

The scientific method description documents the shared data construction,
verified W64 optimization recipe, NW2 adaptation evidence, coarse-to-fine
inference policy, evaluation, and limitations:
[METHOD_DESCRIPTION.md](METHOD_DESCRIPTION.md).
