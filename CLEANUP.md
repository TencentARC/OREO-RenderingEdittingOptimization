# OREO release cleanup ledger

This ledger tracks the reduction of the internal research snapshot into the
minimal OREO code release. It records intentional removals; Git remains the
source of truth for exact file-level diffs.

## Baseline

- Baseline commit: `b593999c2aab50a322b7b8ed590fcfecd73c659d`
- Baseline tag: `pre-public-cleanup`
- Cleanup branch: `cleanup/dual-stage-release`
- Source: clean copy uploaded to the private Gongfeng repository
- Baseline size: 349 tracked files, with no tracked file larger than 50 MiB

The baseline is an internal recovery point, not a public-release approval. It
may still contain internal paths, cluster assumptions, unused research methods,
or third-party components that require review.

## Retained release contract

The release is centered on the Dual-Stage Contrastive FlowEdit algorithm used
by `checkpoint_10_3157`. It must retain:

- the dual-stage configuration, entry point, and launcher;
- both trained TRELLIS stages and dual-weight checkpoint handling;
- the shared Stage 1 and Stage 2 computations required by the dual-stage path;
- the minimum FlowEdit and DINO conditioning implementation used by that path;
- reproducible training, inference, and AlphaImages evaluation instructions.

The standalone Stage-2-only algorithm is not part of the public interface. Its
entry point, configuration, and launcher may be removed, while shared Stage 2
internals required by the dual-stage implementation must remain.

## Removal ledger

Add one row in the same commit as each intentional removal. Use the Git commit
for exact paths and this table for the reason and compatibility impact.

| Commit | Area removed | Reason | Retained behavior / impact |
| --- | --- | --- | --- |
| `cleanup: remove standalone stage-2-only entry points` | Standalone Stage-2-only configuration, training entry point, and multi-node/single-node launchers | The public release exposes only the Dual-Stage Contrastive FlowEdit algorithm; these files duplicated a standalone interface outside that contract | Dual-Stage configuration, entry point, and launchers remain. Shared sparse-stage computation in `autograd_template.py` and operations in `stage_ops.py` remain available to the Dual-Stage path. |
| `cleanup: remove Hunyuan3D generator` | Vendored Hunyuan3D generator implementation and its standalone launchers | Repository-wide consumer tracing found no import or dynamic path from the retained Dual-Stage implementation; the launchers referenced absent training and configuration files | TRELLIS Dual-Stage training, inference, shared stage operations, and evaluation remain. Hunyuan3D-specific `pymeshlab` and `scikit-image` dependencies were removed. |

## Verification ledger

These entries harden or verify the retained release path; they are not code-removal claims.

| Commit | Verification hardening | Evidence / enforced contract |
| --- | --- | --- |
| `fix: require full dual-stage checkpoint loading` | Added an explicit import-light loader for the flat Accelerate Dual-Stage checkpoint and made Dense FlowEdit evaluation fail on incomplete loads or unchanged student weights | `model.safetensors` loads strictly into `slat_flow_model`; `model_1.safetensors` loads strictly into `sparse_structure_flow_model`; both parameter sets must exactly match their corresponding teacher/base keys and shapes and differ above threshold. Eval-only registration preserves sparse → dense Accelerate model order even when the dense optimizer object is absent. The mapping, load counts, and both parameter-difference reports are written to the evaluation metadata. Covered by `python3 -m unittest tests.test_trellis_full_dual_checkpoint`. |
| `fix: define dual-stage single-node run name` | Defined the retained single-node Dual-Stage launcher's run name before its first use | The launcher defaults to `trellis_dualstage_contrastive_single`, preserves a caller-supplied non-empty `RUN_NAME`, and passes the value to the config as a quoted argument. |
| `docs: record full dual-stage smoke verification` | Ran an isolated one-GPU smoke on the existing H200 Debug Pod with commit `3c5e733966aef18c8755c6e17c78633af85d877f` and `checkpoint_10_3157` | Result: `PASS`. The strict load report records 526 tensors / 600,432,520 elements from `model.safetensors` into `slat_flow_model`, and 489 tensors / 559,737,864 elements from `model_1.safetensors` into `sparse_structure_flow_model`. Student/base comparison found 526/526 differing slat parameters (`max_abs_diff=0.0697021484375`) and 488/488 differing sparse-structure parameters (`max_abs_diff=0.02215576171875`), with exact key and shape coverage. Eight non-empty teacher outputs and eight non-empty student outputs were generated. Artifacts are isolated at `s3://arcwm-code-us-west-2/ericzyma/oreo-verification/debug-full-dual-smoke-3c5e733-20260824-002000/`, including `smoke_summary.json` and `flowedit_params.json`. |
| `docs: record single-GPU dual-stage training smoke` | Resumed `checkpoint_10_3157` on one H200 with one image / one batch, both full-parameter optimizers enabled, evaluation and visualization disabled, and no reduced-resolution or reduced-step algorithm overrides | Result: `PASS` at epoch 11 / global step 3158. The saved `checkpoint_11_3158` contains both model and optimizer files. Comparing before and after with exact key/shape coverage found all 526 `model.safetensors` tensors changed (`max_abs_diff=0.000152587890625`) and 488/489 `model_1.safetensors` tensors changed (`max_abs_diff=0.00006407499313354492`). `train.csv` contains one row with finite sparse and `dense/` contrastive losses. The eight-rank checkpoint's random state could not be restored on the one-rank topology, so this proves model/optimizer resume and both student updates, not bitwise stochastic continuation. Evidence is isolated at `s3://arcwm-code-us-west-2/ericzyma/oreo-verification/debug-single-gpu-train-smoke-3c5e733-20260824-004000/`. |
| `fix: pin compatible Qwen environment` | Aligned the declared and Koala environments on the Qwen Image Edit compatibility tuple and moved runtime inspection to an explicit standalone diagnostic | Python 3.12, PyTorch 2.6.0+cu124, Diffusers 0.38.0, Transformers 4.57.6, Tokenizers 0.22.0, Accelerate 1.4.0, Safetensors 0.8.0, Flash Attention 2.7.3, and Spconv cu124 2.3.8 are pinned. Koala uses a new immutable cache identity and never silently restores the legacy Transformers 5 cache. The import-light contract is covered by `python3 -m unittest tests.test_qwen_environment_contract`; detailed runtime inspection remains opt-in through `scripts/test_qwen_environment.py` and is not part of the training entry point. |
| `fix: make FlowEdit source unconditional` | Removed the Source prompt and Source CFG surface from Qwen FlowEdit while retaining Target CFG | Source reuses the Target CFG unconditional/negative embedding and performs exactly one unconditional transformer evaluation per timestep; it no longer computes a Source conditional prediction, Source CFG combination, or Source norm rescaling. Target conditional/unconditional evaluation and `true_cfg_scale_tgt` remain unchanged. Direct configurations and callers no longer expose `source_prompt`, `negative_prompt_src`, or `true_cfg_scale_src`. Covered by `python3 -m unittest tests.test_flowedit_source_unconditional`. The separate TRELLIS 3D FlowEdit evaluation rollout is outside this Qwen training path and retains its evaluation semantics. |
| `docs: record compatible Qwen environment cache` | Rebuilt the complete Koala Python environment under the new immutable cache identity and verified a fresh restore | Starting from the complete legacy CUDA environment, Transformers 4.57.6 and Tokenizers 0.22.0 were installed; Hugging Face Hub resolved to 0.36.2. `uv pip check` passed for 197 packages. Before upload and after an isolated S3 restore, exact versions were confirmed for Python 3.12, PyTorch 2.6.0+cu124, Torchvision 0.21.0+cu124, Diffusers 0.38.0, Transformers 4.57.6, Tokenizers 0.22.0, Accelerate 1.4.0, Safetensors 0.8.0, Flash Attention 2.7.3, and Spconv cu124 2.3.8. On the CPU Debug Pod, QwenImageEditPlusPipeline and the retained CUDA extension modules imported successfully; CUDA kernel execution, Qwen inference, and training were not run. The 9,476,044,800-byte object is isolated at `s3://arcwm-code-us-west-2/ericzyma/data/flow_grpo_v2/envs/uv-py312-torch260cu124-diff0380-tf4576-v1.tar`. The legacy `uv-venv.tar` was not overwritten. |
| `docs: record Source-unconditional GPU smoke` | Ran an instrumented real-weight Qwen FlowEdit step on one H200 with commit `ff02c71b58916530eb38f923b2f3f81bb2e1ed17` and the pinned environment | Result: `PASS` with a two-timestep scheduler and `n_max=1`, so exactly one FlowEdit step executed. The observed transformer cache-context sequence was Source `uncond` → Target `cond` → Target `uncond`; both unconditional calls reused the same prompt-embedding storage, all three outputs were finite, and Target conditional/unconditional `x0` differed (`L2=0.8991051316261292`). The 1024×1024 output was non-empty (`pixel_std=53.53466033935547`) and peak allocated GPU memory was 57.92431592941284 GiB. Evidence is isolated at `s3://arcwm-code-us-west-2/ericzyma/oreo-verification/debug-source-uncond-ff02c71-20260824-145714/results/`. The cached Qwen tar contained blobs and `refs/main` but lacked snapshot links; Hugging Face metadata was required to reconstruct snapshot `6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9`, after which the smoke loaded that exact local snapshot. The first attempt with a one-timestep scheduler correctly failed finite-output checks because that scheduler configuration divided by zero. No training step or full Dual-Stage training run was performed. |
| `fix: require exact offline Qwen snapshot archive` | Added a 35-file manifest with expected sizes and content digests, plus a verified archive build/extract helper and fail-closed Koala setup paths | Snapshot files must resolve to the pinned `6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9` blobs; SHA-1 Git blob and SHA-256 LFS content digests reject same-size corruption, missing/extra snapshot entries, wrong refs, broken links, and archive path traversal. The immutable Qwen URI is `s3://arcwm-code-us-west-2/ericzyma/data/flow_grpo_v2/models/qwen-image-edit-2511-6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9-hf-v1.tar`; `--fast` is offline and never falls back to the legacy Qwen tar, `--qwen-only` restores only the environment and Qwen, and bare `--fast` retains TRELLIS, dataset, and DINOv2 restoration. Covered by `python3 -m unittest tests.test_qwen_environment_contract tests.test_qwen_snapshot_archive`. |
| `docs: record exact Qwen archive build` | Built and independently verified the pinned Hugging Face cache archive on Koala | Job `ericzyma-oreo-qwen-arc-v1-normal-20260824-171425`: `Succeeded`, `QWEN_ARCHIVE_PASS`. Archive `s3://arcwm-code-us-west-2/ericzyma/data/flow_grpo_v2/models/qwen-image-edit-2511-6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9-hf-v1.tar` is 57,718,927,360 bytes with SHA-256 `ea8bcff5e1fdce8c8d17ebf153f6109d7e97d54ffb757c86ffbaefd2fec01758`; commit `6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9`, manifest 35. Source cache, archive contents, and isolated extraction all passed the exact content-digest contract. Results: `s3://arcwm-code-us-west-2/ericzyma/oreo-verification/qwen-archive-build-20260824-171130/results/`. |
| `docs: record Qwen fast restore verification` | Restored the pinned environment and Qwen archive through the real qwen-only fast path from clean local state | Job `ericzyma-oreo-qwen-fast-v1-normal-20260824-174021`: `Succeeded`, `QWEN_FAST_RESTORE_PASS`, 0 GPU / 16 CPU / 64 GiB, 717 seconds. The job executed `scripts/setup_koala.sh --fast --qwen-only` with strict offline snapshot resolution; Python 3.12.3, PyTorch 2.6.0+cu124, Torchvision 0.21.0+cu124, Diffusers 0.38.0, Transformers 4.57.6, Tokenizers 0.22.0, Accelerate 1.4.0, Safetensors 0.8.0, Flash Attention 2.7.3, and Spconv cu124 2.3.8 matched the pinned tuple. Evidence: `s3://arcwm-code-us-west-2/ericzyma/oreo-verification/qwen-fast-restore-20260824-173839/results/qwen_fast_restore_summary.json`. This verifies restore and imports only; it is not GPU inference or a full tensor load. |

## Planned cleanup batches

These are candidates, not proof that the files are safe to delete. Each batch
must trace imports, configuration dispatch, dynamic entry-point loading, tests,
and documentation before removal.

1. Remove unrelated experimental algorithms such as distillation, GAN,
   latent-GAN, tri-image, bilevel, hybrid, normal-similarity, GRPO, and SD3
   paths when they have no consumer in the retained dual-stage path.
2. Remove the standalone Stage-2-only configuration, entry point, and launcher
   without deleting its shared computation used by Dual-Stage Contrastive
   FlowEdit.
3. Remove datasets, checkpoints, generated results, debug artifacts, obsolete
   tests, and machine-local setup helpers that are not part of reproduction.
4. Replace or parameterize internal filesystem, S3, Koala, and service-specific
   assumptions needed by retained code.
5. Reduce dependencies to those exercised by installation, inference,
   evaluation, and training verification.
6. Add release documentation, third-party notices, and checkpoint/data download
   instructions after the retained implementation is verified.

## Commit policy

- Keep each cleanup commit focused on one removal category.
- Do not rewrite or force-push the baseline.
- Do not commit credentials, model weights, datasets, logs, or generated output.
- Preserve the original research repository and its uncommitted files.
- Verify the retained dual-stage entry path after every structural batch.
- Record failures and deferred uncertainties instead of claiming unsupported
  reproducibility.

## Publishing status

- Private Gongfeng: baseline and reviewed verification fixes are uploaded on
  `cleanup/dual-stage-release`; no merge to `main` has been performed.
- Public GitHub: intentionally deferred.
