# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
poetry install

# Run a script against the package
poetry run python examples/validate.py

# Run a one-off script with uv (preferred for standalone scripts)
uv run script.py
```

There are no tests or linting configured yet.

## Architecture

Oval is a Python library for analyzing vote data from computational democracy platforms (e.g., Polis). The core abstraction is:

1. **`Conversation`** (`conversation.py`) — holds `Comment`, `User`, and `Vote` objects and lazily builds a `votes_matrix` (users × comments numpy array, with NaN for no vote). Caches index lookups for users and comments.

2. **`io.read_polis`** (`io.py`) — reads Polis CSV exports (comments + participant-votes files) into a `Conversation`. The votes CSV must have comment IDs as column headers.

3. **`decompose_votes`** (`decomposition.py`) — reduces the vote matrix to lower-dimensional space using PCA (default) or UMAP. Scales the output by the square root of each participant's vote count.

4. **`Variable`** (`variable.py`) — abstract base for predicting a latent variable (e.g., a rating dimension) across all comments. Two implementations:
   - **`LinearVariable`**: projects participants into PCA/UMAP space, fits a linear model from anchor comment votes → anchor scores, then predicts non-anchor comments by dot product of votes with participant predictions.
   - **`DiffusionVariable`**: builds a bipartite participant–comment graph and runs label propagation (personalized PageRank-style) from anchor comment scores outward. Normalizes anchors and non-anchors separately.

   Both implement `fit(anchors: dict[int, float])` (comment_id → score) and `predict_comments(comment_ids)`. `score_comments` (on the base class) computes Pearson correlation against held-out anchors.

**Typical usage pattern:**
```python
from oval.io import read_polis
from oval.variable import DiffusionVariable

conversation = read_polis("comments.csv", "participant-votes.csv")
variable = DiffusionVariable(conversation=conversation)
variable.fit(anchors={comment_id: score, ...})
r = variable.score_comments(test_anchors)
```

Note: the `oval/` package has no `__init__.py`, so submodules must be imported directly (`from oval.io import ...`).
