# Section 4: deterministic d1-3B conversion

This stage converts existing partitions to string `prompt` / `completion` examples on CPU. It does not initialize a model, run inference, initialize TRL, configure LoRA, or train. Python 3.12, Transformers 5.19.0 and Tokenizers 0.23.3 were exercised locally without torch installed. All dependencies are locked in `uv.lock`.

## Reference and pinned assets

Inspected the [BANKING77 notebook](https://github.com/Liquid4All/cookbook/blob/main/finetuning/notebooks/sft_for_decision_models_with_trl.ipynb) and the actual [d1 prompt utility at the pinned commit](https://huggingface.co/LiquidAI/d1-3B/blob/051bcc464b01b9f92942b364d9586b0ef5912432/prompt.py) before implementation. Model repository revision: `051bcc464b01b9f92942b364d9586b0ef5912432`. The config records expected hashes for exactly five files: `prompt.py`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, and `LICENSE`. Download requests explicitly allow only these names. The first run needs network access for assets; later runs default to offline and fail on missing/tampered files. No full snapshot or model weights are downloaded. The prompt utility uses only the Python standard library and is imported only after hash verification; tokenizer loading disables remote code.

## Example construction

1. Verify saved split/source hashes, configuration, partition completeness, group separation, and row equality with the processed cohort. Keep each experiment and each partition separate.
2. Construct a dictionary in the existing 85-feature allowlist order. Preserve field names, integer types, and full Python float precision; JSON roundtrips numeric values without decimal rounding. Fail on missing/nonfinite values or special-token markup. Ignore every unlisted field. UUID is traceability metadata, not a feature.
3. Always include `Binder` and `Non-binder`. Shuffle the two with `random.Random("decision-binder-v1:<seed>:<uuid>")`. This is independent of the experimental label, input row order, and split name, so the inference formatter can reproduce the identical choice list without knowing the answer. A seed change intentionally changes option order.
4. Call `as_question()` with a choice question, the fixed instruction “Is this candidate expected to bind its intended protein target?”, and the ordered criteria. Call `aliases()` to obtain distinct single-token codes for the ordered choices. Never hard-code A as binder: the ordering varies per UUID.
5. Call `render()` with the state dictionary, the question, and the tokenizer BOS. Explicitly use `json_only`, no system turn, no reasoning lead, and the native description-style choices. The prompt opens a user turn, lists state and question, closes that turn, and opens the assistant answer slot.
6. Map experimental 1 to `Binder` and 0 to `Non-binder`; completion is that option's alias plus the tokenizer EOS. At this revision codes are A/B (token IDs 41/42) and EOS is `<|im_end|>` (124900). Verify these via the tokenizer rather than assuming them.

This follows the notebook's model-native rendering and answer-code-plus-EOS convention. Adaptations: with only two classes there is no 27–77-option subsampling; both are always present. UUID-based seeded shuffling replaces split/index seeding and is explicitly label-independent. We load only lightweight assets, not the notebook's whole model snapshot. No generic chat template is applied on top of the rendered strings.

## Why the native format matters

d1's decision interface reads option-token scores at a particular assistant answer slot. Its prompt utility defines that slot, role boundaries, state/question layout, and tokenizer-compatible aliases. An answer token means an option position in that example, not a fixed biological class. The training example must associate the correct position with the native rendered input. The same `build_prompt()` function is available for later inference and does not require a label; it returns the prompt, state, and option encoding. No inference is implemented here.

## Saved artifacts and deterministic validation

`data/sft/<mode>/dataset/` is a `DatasetDict` saved with `save_to_disk`; the three split Parquets contain the same `uuid`, `prompt`, and `completion` fields. UUID may be removed from a trainer-facing copy later, while retaining the original for audit. No sequence, target label, experimental measurement, or split identifier is inserted into the state unless it is an explicitly permitted computational feature (the target protein name itself is allowed).

The input frames are sorted by UUID. No wall-clock timestamp is part of conversion outputs. Fixed dependency versions, source checksums, utility checksums, seed, feature order, and output hashes are recorded in `results/sft/<mode>/report.json`. Three examples, one per partition, are saved in `examples.json` and a readable primary `examples.md`. They include allowed original features, the complete prompt, label-to-code mapping, completion, and token counts. They are review artifacts, not tuning data; inspecting them does not authorize selecting model parameters on test examples.

Every row is checked for answer-code token length and decode roundtrip, exactly two completion tokens (answer and EOS), stable tokenization of `prompt + completion` versus the concatenated individual encodings, one leading BOS, the assistant suffix, and no extra special tokens under the tokenizer's default encoding. Full strings are saved; there is no truncation. Tokenizer `model_max_length` is an uninformative sentinel here, so measured lengths are reported instead of claiming a model context guarantee.

Tests also show that flipping labels or adding outcome columns cannot change the prompt; both option orders occur; row ordering does not change conversion; training and inference rendering agree; the actual pinned tokenizer works without torch; and saved datasets preserve the source partitions. An absent local tokenizer causes tokenizer-specific tests to skip in a fresh checkout: run the documented asset download first for the full suite.

## Deferred TRL completion-only loss verification

At Section 4, tokenizer validation was complete and trainer/collator checks were deferred. Section 6 has now verified actual TRL-prepared labels and collator masks on all train/validation rows; CUDA forward/backward verification remains pending. See [training status](training.md). Before later training, pin and inspect the chosen TRL version and explicitly enable completion-only loss for the string prompt/completion dataset. Use only the training partition for weight updates, validation for evaluation during development, and keep test out of trainer fitting and hyperparameter selection.

Before the first optimizer step, inspect a small collated batch containing both answer codes and unequal prompt lengths:

- Verify prompt token labels are all `-100`, including BOS, user content, question, and assistant header.
- Verify the only non-padding supervised labels are the intended answer token and the EOS token; padding must also be `-100` and have the correct attention mask. Account for the causal language model's internal one-token shift when checking predictions against labels.
- Decode the supervised token IDs and compare exactly with the stored completion. Check that no additional chat formatting, duplicated BOS/EOS, prefix retokenization, or label-dependent prompt transformation occurred.
- Set any future length limit from the measured full-example lengths and confirm no completion/EOS was truncated. Recheck masking if packing is enabled; initially inspect un-packed batches.
- Verify the dataset handed to training contains only train UUIDs. Preserve UUID lookup separately if the trainer strips non-training columns.

These were the acceptance checks planned at Section 4. Section 6 now implements the trainer, LoRA configuration, and Colab wrapper and has verified masking on CPU; it has not run optimizer steps. The Section 4 conversion itself still introduces none of those training actions.
