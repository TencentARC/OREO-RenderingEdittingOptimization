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
