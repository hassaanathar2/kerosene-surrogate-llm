# Changelog

## 1.0 (current)

Four independent, self-contained, single-purpose scripts, one per
representation (composition-only, ChemBERTa, MolFormer, TinyLlama), each
defining its own copy of the same regression head and evaluation protocol
(kept identical on purpose — see `tests/test_scripts_consistency.py`). This
release fixes four issues found in an earlier internal version of this
pipeline:

- **Mixture-embedding averaging.** Component embeddings were previously
  combined into a single composition-weighted mixture embedding
  (`E_mix = sum_i x_i * E_i`) before being fed to the regression head. This
  collapses the per-component structural information the embeddings were
  meant to provide down to a linear function of composition. Component
  embeddings are now kept separate, each reduced with its own PCA, and paired
  with its mole fraction as a per-component input slot:
  `X = [x_1 ... x_N, x_1*z_1, ..., x_N*z_N]`.

- **PCA fit on the wrong axis.** The PCA step in the ChemBERTa/MolFormer path
  is fit on the (small) set of pure-component embeddings, not on samples,
  so it cannot leak test-set information. The PCA step in the TinyLlama path
  (which embeds one prompt per sample) is fit on the training split's prompt
  embeddings only, rather than on the full dataset before splitting.

- **Validation using the test set.** Early stopping previously monitored
  loss on the test set. It now uses a validation split carved out of the
  training data only (15%, fixed `random_state=0`); the test set is used
  exactly once, after training is complete.

- **Inconsistent regression heads.** Each representation previously used a
  different head (different depths, different regularisation, one with batch
  normalisation and a learning-rate scheduler, one without). All four scripts
  now train the identical 128-64 MLP (dropout 0.10, AdamW, lr 1e-3, weight
  decay 1e-4, batch size 64, patience 80), so that differences between
  representations are not confounded by differences in the regressor.
  `tests/test_scripts_consistency.py` checks that the four scripts' copies of
  this head, the column auto-detector, and the metrics function stay
  identical.

None of the four scripts read the test set during PCA fitting, scaling, or
feature construction.
