# Attribution and licensing review

Reviewed 2026-10-28. Licensing of the dataset, model, reference code, and this project's original code are separate questions. This document records source terms; it does not grant new rights or certify a particular deployment's compliance.

## Dataset and derived records

Source: Anthropic, *Claude protein binder design (data release v1.0)*, including the credited experimental/data contributors and Adaptyv binding calls. Preserve the dataset card's supplied attribution and contributor information when redistributing derived records.

- [Pinned dataset card](https://huggingface.co/datasets/Anthropic/claude-protein-binder-design/blob/9e1b81696da46835e9e9cde9a3da976e0abc92ab/README.md), revision `9e1b81696da46835e9e9cde9a3da976e0abc92ab`, declares **CC BY 4.0**.
- [CC BY 4.0 terms](https://creativecommons.org/licenses/by/4.0/) require appropriate credit, a license link, and indication of modifications; attribution must not imply endorsement. Preserve supplied notices and do not impose additional restrictions on licensed material.
- Our modifications: select primary target forms and prescribed stoichiometries, aggregate Boltz-2 metrics, exclude unresolved outcomes/incomplete aggregate records, group/split designs, render prompts, and add model probabilities. Prediction and audit records are derived data, even though they are small enough to retain in Git.

Suggested write-up attribution: “Uses the Anthropic Claude protein binder design data release v1.0 (revision 9e1b81696da46835e9e9cde9a3da976e0abc92ab), CC BY 4.0. We applied the filtering, aggregation and split procedures documented in this repository. Experimental binding labels come from the release’s Adaptyv calls.” Include links above and retain the release's contributor notices.

## Model and adapters

Source: Liquid AI, **LiquidAI/d1-3B**, revision `051bcc464b01b9f92942b364d9586b0ef5912432`.

The pinned [model LICENSE](https://huggingface.co/LiquidAI/d1-3B/blob/051bcc464b01b9f92942b364d9586b0ef5912432/LICENSE) is **LFM Open License v1.0**, not Apache/MIT. Its local checksum-verified copy was inspected; SHA-256 `4d28ca14dedc0b3d0fcc2b3339f0e79931faa33874f3d24f522183a8fc70068c`. The web viewer was unavailable during this audit; the reviewed text is the pinned downloaded license.

Redistribution conditions include providing the license, marking modified files, retaining relevant copyright/patent/trademark/attribution notices, and retaining applicable NOTICE attribution. Section 5 limits commercial use by reference to a US$10 million annual-revenue threshold; consult its exact definitions and nonprofit/research provision for the intended entity/use. Do not describe the model as unrestricted open source. Check these terms before distributing an adapter or merged model; a small adapter is not automatically exempt from upstream requirements. We keep all model/adapter weights out of Git. The repository does not decide whether a particular organization qualifies for commercial use.

## Reference methods and code

- Colby T. Ford: [Zero-Shot Classification of Protein Binders with Jev](https://github.com/colbyford/jev-binder-classification), especially its Adaptyv preprocessing notebook, and the [accompanying article](https://blog.colbyford.com/zero-shot-classification-of-protein-binders-with-jev-634b0300e58d). Credit the preprocessing inspiration, not an alleged reproduction of its model-performance numbers.
- Liquid4All: [BANKING77 SFT decision-model tutorial](https://github.com/Liquid4All/cookbook/blob/main/finetuning/notebooks/sft_for_decision_models_with_trl.ipynb). Credit its native prompt/answer conventions, TRL completion-only SFT, and LoRA configuration. Our notebook is a project execution wrapper, not a redistributed copy of the tutorial.
- Native `prompt.py` and model code are downloaded at the pinned model revision, hash checked, and kept in ignored storage. They are not relicensed by this repository.

A reusable-code license for the two reference notebook repositories was not established by this review (the cookbook root LICENSE URL returned 404; no license was shown in the inspected Ford root listing). Public visibility alone is not redistribution permission. Verify the specific source version's terms before copying/redistributing notebook text or code. Historical notebook commit IDs were not recorded; the dataset/model revisions are pinned, but reference links may change.

## Repository code

No project-wide license has been selected for the original code. The owner must choose it before advertising unrestricted code reuse; this review does not silently assign one. Any chosen project license must preserve independent upstream data/model terms. `uv.lock` records dependency identities/versions; those packages retain their own licenses.
