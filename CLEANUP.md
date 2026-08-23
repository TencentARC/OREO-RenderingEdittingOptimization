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

- Private Gongfeng: baseline uploaded; cleanup branch is being prepared.
- Public GitHub: intentionally deferred.
