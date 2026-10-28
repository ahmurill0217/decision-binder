# Section 6: TRL supervised fine-tuning with LoRA

**Status:** implementation and real-model CPU preflight are complete. GPU execution was left for Colab at the user's request. No optimizer steps have run, no trained adapter exists yet, and neither a T4 smoke pass nor its peak memory usage is claimed. Full training is disabled by default.

## Files and installation

- `src/decision_binder/train.py`: data checks, real TRL mask checks, PEFT setup, CUDA hardware checks, smoke/full training, adapter save/reload, metadata.
- `scripts/04_train_lora.py`: thin CLI; defaults to CPU-only `preflight`.
- `configs/experiment.yaml`: all training hyperparameters and smoke limits.
- `notebooks/finetune_colab.ipynb`: repository clone/install/data preparation and CLI execution wrapper.

```bash
uv sync --frozen --extra dev --extra training
uv run --frozen --extra training python scripts/04_train_lora.py --mode preflight --download-model
```

The CPU preflight loads the original weights and LoRA layers but runs no model forward, backward, optimizer, or validation-loss calculation. It works on the local Apple Silicon machine; only smoke/full modes require CUDA. MPS training is deliberately not enabled. All dataset access in the training module is limited to the train and validation Parquets. SFT and split manifests are checked, UUID overlap fails, and changing the max length cannot silently truncate the answer.

## LoRA and reference adaptations

Base: `LiquidAI/d1-3B`, immutable revision `051bcc464b01b9f92942b364d9586b0ef5912432`. The implementation follows the [Liquid BANKING77 notebook](https://github.com/Liquid4All/cookbook/blob/main/finetuning/notebooks/sft_for_decision_models_with_trl.ipynb), with Transformers 5.19.0, TRL 1.15.0, and PEFT 0.21.2. Dependencies are an optional `training` extra so dataset preparation does not require them. The reference's unused torchao install is omitted; no quantization, QLoRA, or architecture changes are introduced.

LoRA rank is **16**, alpha **32**, dropout **0.05**, bias `none`, task `CAUSAL_LM`. It adapts `q_proj`, `k_proj`, `v_proj`, `out_proj`, `in_proj`, `w1`, `w2`, and `w3`, covering language attention, convolution projections, and MLPs. The regex `.*vision_tower.*` excludes vision modules with matching names. All original weights are frozen. The actual model exposes **24,461,312 trainable parameters** out of **3,147,945,200**, approximately **0.777%**, matching the reference. Preflight checks every trainable name is LoRA-only and not in the vision tower and verifies each configured target module is represented.

Initial full-run settings: one epoch, learning rate 2e-4, cosine schedule, 5% warmup, gradient checkpointing, maximum length 2048, no packing. The following adjustments are explicit:

- Microbatch **1**, accumulation **16**, instead of reference 4 × 4. Effective batch remains 16 on one GPU; our structured prompts reach 1,523 tokens, so a smaller microbatch leaves more room on a T4. Validation batch is 1.
- Max length 2048 is explicit. TRL's current default 1024 would truncate these examples. Every full token sequence is checked before trainer construction; every prepared sequence is checked again against the source.
- Non-reentrant gradient checkpointing is explicit; checkpoint flags are checked on the actual model layers.
- `adamw_torch` is explicit for portability. No 8-bit optimizer is used.
- Evaluate/save every 25 steps, retain two checkpoints, log every 10 steps. The one-epoch run has roughly 48 optimizer steps; the reference's evaluation interval of 50 would otherwise miss intermediate validation. Final validation always runs. No automatic best-checkpoint selection is enabled.

These are initial settings, not settings selected from test performance. Smoke settings use 8 training examples, 4 validation examples, 2 optimizer steps, accumulation 1, and logging/evaluation each step. The deterministic subsets include the longest example in each eligible split to exercise realistic sequence memory. Smoke steps are capped at five in code. Smoke adapters are never reused to initialize the full experiment.

## What TRL sees and how loss is computed

The trainer receives ordinary string `prompt`/`completion` datasets and the pinned tokenizer via `processing_class`; UUIDs are kept in audit metadata but removed from trainer-facing inputs. There is no additional chat formatting or custom loss. `completion_only_loss=True`, `assistant_only_loss=False`, `packing=False`, and `padding_free=False` are explicit.

In TRL 1.15.0, dataset preparation tokenizes the prompt and combined prompt/completion and builds labels with `-100` on prompt positions. The actual collator right-pads examples and assigns `-100` to padding. The two remaining supervised tokens are the answer code and `<|im_end|>`. Causal shifting makes the assistant answer-slot hidden state predict the code, then the following hidden state predict EOS. Loss is the mean negative log probability over those supervised tokens, accounting for accumulation; prompt tokens provide context but contribute no direct loss. The pinned trainer uses its native fused LM-head loss path. CUDA forward/backward compatibility still requires the smoke run.

**Actually verified on CPU:** TRL-prepared token IDs exactly match the source for all **754 train and 161 validation records**, no examples were dropped, each row has exactly two supervised tokens, and prompt/padding masks are correct. Collated shortest/longest examples and both completion codes were inspected. This verifies real dataset preparation and masking, not finite GPU training loss.

## CUDA precision and memory

The recommended reference is an NVIDIA T4 (16 GB marketed, approximately 14.6 GiB reported). Before loading weights, the CLI displays/records GPU identity, compute capability, total/free memory, and selected precision. It requires at least 14 GiB total and 10 GiB free memory. `auto` selects native bf16 where supported and fp16 on T4; unsupported explicit bf16 fails. No CPU/MPS fallback is made for a requested training run.

Unquantized base weights alone need roughly 6 GB in half precision. LoRA weights, gradients, optimizer state, activations, temporary attention/loss buffers, allocator overhead, and checkpoints add memory. The total cannot be certified from parameter count. Microbatch 1 and checkpointing reduce activation pressure; actual CUDA peak allocated/reserved memory is measured and saved by the smoke test. **T4 peak memory remains unmeasured here.** If smoke fails with OOM, stop and review its metadata/runtime rather than silently switching methods or claiming compatibility. Checkpoint storage and free disk are separate from GPU memory; allow room for weights and optimizer checkpoints.

## Colab smoke test

1. Publish this project to a Git repository. This local checkout currently has no Git remote; no repository URL has been invented or published by the implementation.
2. Open `notebooks/finetune_colab.ipynb` in Colab. Set `REPO_URL`, preferably set `REPO_REF` to an immutable commit, and select an NVIDIA GPU runtime with Python >=3.12.
3. Leave `RUN_FULL_TRAINING = False`. Enable Google Drive for persistent output, or download results before the ephemeral runtime disconnects.
4. Run the cells. They clone the repository, install the locked training extra, display `nvidia-smi` and PyTorch GPU details, generate missing prepared datasets using existing scripts, and execute the Python smoke mode.
5. Review `metadata.json`, printed train/validation loss, measured GPU memory, and the saved adapter. `status: passed` requires finite losses, finite changed LoRA weights, successful adapter saving, and exact tensor equality after reload into a fresh base model. An initial `preflight_passed` status does **not** mean the GPU smoke passed.

The notebook generates datasets only if needed. The preparation pipeline may generate test artifacts as part of reproducible splitting; the training script never opens them. It does not run the baseline evaluator or any new evaluation method. Notebook Python cells were syntax-checked locally; no Colab cells have been executed here.

Equivalent CUDA CLI smoke command:

```bash
uv run --frozen --extra training python scripts/04_train_lora.py \
  --mode smoke --download-model --output-dir /path/to/persistent/smoke-run
```

## Full training — instructions only, not executed

After reviewing a successful GPU smoke and explicitly approving the full run:

```bash
uv run --frozen --extra training python scripts/04_train_lora.py \
  --mode full --approve-full-run --output-dir /path/to/persistent/full-run
```

In Colab, setting `RUN_FULL_TRAINING = True` is the corresponding explicit opt-in. Its guarded cell requires the smoke report to have passed. Full mode starts fresh from the same pinned base and a newly initialized adapter. The CLI rejects full mode without its approval flag and refuses nonempty output directories to protect earlier runs. Automatic checkpoint resume is not implemented; saved checkpoints preserve state for a separately reviewed recovery if needed.

Artifacts include `adapter/` (PEFT adapter, config, tokenizer), trainer state/checkpoints, `progress.json`, and `metadata.json` (parameters, exact UUIDs used, source/model/code hashes, package versions, hardware, mask verification, losses, adapter changes, and reload checks). The adapter config names the upstream model and pinned revision rather than a Colab-local base path; no merge into base weights is performed. Persistent Drive storage is written directly rather than waiting until the final cell.

Current saved artifact: `results/training/primary_preflight/metadata.json`. It documents CPU preflight only. GPU smoke loss, trained adapter, reload evidence, and T4 memory measurements remain pending execution in Colab. Full training has not been initiated.
