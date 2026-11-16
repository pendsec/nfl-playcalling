"""
Neural outcome models (PyTorch) — TARNet and DragonNet over the pre-snap state.

Both learn one representation of the state, phi(s), shared by every call, and
read Q(s, a) off it with one head per call (Shalit et al. 2017; Shi et al.
2019). Sharing the representation is what a regression Q gets from pooling: a
rarely called alignment borrows the state structure every other call learned,
and only its head — a small MLP on phi — is its own.

  * TARNet — phi(s) -> head_a(phi) for each call a; trained on the factual
    reward only (each snap's loss uses the head of the call actually made).
  * DragonNet — TARNet plus a propensity head pi(a | phi) trained alongside,
    so the representation must keep what predicts the CALL — the confounders —
    rather than only what predicts the reward. And a targeted-regularization
    term (the TMLE fluctuation built into the loss): a learnable epsilon_a per
    call, and the extra loss

        (R - Q(s, A) - epsilon_A / pi(A | s))^2

    whose stationarity in epsilon makes the fitted Q satisfy the doubly-robust
    estimating equation — the multi-call version of the original binary term.

Heads (`outcome_model.neural.heads`):
  * factored (default) — one head over (phi, the call's package and front, its
    one-hot): the neural counterpart of the pooled GBM, so a rare cell shares
    strength with every call that has its package or its box.
  * separate — one small MLP per call, as in the original architectures. Each
    call's head is fit with independent noise, which inflates the spread
    across calls (1.4-1.9x the truth on the synthetic league) — the per-action
    optimizer's curse again (experiments/outcome_models/).

The factored DragonNet is the neural Q candidate: on the real holdout it ties
the pooled GBM on doubly-robust loss. Its game-bootstrap spread is too tight for
the safe policy as it stands (it departs ~2.5x as often), so it needs its own
uncertainty estimate before it can be gated against the GBM.

The state goes in as standardized numerics plus learned embeddings for the
categoricals (defense, offense, formation, QB alignment); index 0 of every
embedding is reserved for a category not seen in training and stays at zero.

Causal note: the networks see the SCM adjustment set only (checked by
`q_model.fit_q_model`); the unobserved read biases them exactly as it does the
GBM. Same interface as `QModel`, so policies, the bootstrap ensemble and OPE
take them unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch import nn


from ...schemas.dataset import Dataset

ARCHITECTURES = ("tarnet", "dragonnet")

# torch and LightGBM each ship their own OpenMP runtime on macOS; once torch has
# run a multi-threaded op, the next LightGBM fit in the process deadlocks. The
# networks are small enough that one thread costs ~30% per epoch, so torch is
# kept single-threaded for the whole process, before any op runs.
torch.set_num_threads(1)


# ── State encoding ───────────────────────────────────────────────────────────
@dataclass
class StateEncoder:
    """Standardize numerics; map each categorical to 1..k (0 = unseen)."""
    numeric: list[str]
    categorical: list[str]
    mean: np.ndarray = field(default=None)
    std: np.ndarray = field(default=None)
    vocab: dict[str, dict] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame) -> "StateEncoder":
        x = df[self.numeric].to_numpy(float)
        self.mean, self.std = x.mean(0), x.std(0) + 1e-6
        self.vocab = {c: {v: i + 1 for i, v in enumerate(sorted(df[c].astype(str).unique()))}
                      for c in self.categorical}
        return self

    @property
    def cardinalities(self) -> list[int]:
        return [len(self.vocab[c]) + 1 for c in self.categorical]

    def transform(self, df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        num = (df[self.numeric].to_numpy(float) - self.mean) / self.std
        cat = np.column_stack([df[c].astype(str).map(self.vocab[c]).fillna(0).to_numpy(int)
                               for c in self.categorical]) if self.categorical else \
            np.zeros((len(df), 0), int)
        return torch.as_tensor(num, dtype=torch.float32), torch.as_tensor(cat, dtype=torch.long)


# ── Networks ─────────────────────────────────────────────────────────────────
def _mlp(sizes: list[int], dropout: float) -> nn.Sequential:
    layers = []
    for i, o in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(i, o), nn.ELU(), nn.Dropout(dropout)]
    return nn.Sequential(*layers)


class Representation(nn.Module):
    """phi(s): embeddings for the categoricals, concatenated with the numerics."""

    def __init__(self, n_numeric: int, cardinalities: list[int], hidden: list[int],
                 dropout: float, embed_max: int = 8):
        super().__init__()
        self.embed = nn.ModuleList(nn.Embedding(k, min(embed_max, (k + 1) // 2), padding_idx=0)
                                   for k in cardinalities)
        width = n_numeric + sum(e.embedding_dim for e in self.embed)
        self.body = _mlp([width] + hidden, dropout)
        self.out_dim = hidden[-1]

    def forward(self, num: torch.Tensor, cat: torch.Tensor) -> torch.Tensor:
        parts = [num] + [e(cat[:, j]) for j, e in enumerate(self.embed)]
        return self.body(torch.cat(parts, dim=1))


class SeparateHeads(nn.Module):
    """One small MLP per call, run as a single batched contraction."""

    def __init__(self, d: int, n_actions: int, hidden: int):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(n_actions, d, hidden) * d ** -0.5)
        self.b1 = nn.Parameter(torch.zeros(n_actions, hidden))
        self.w2 = nn.Parameter(torch.randn(n_actions, hidden) * hidden ** -0.5)
        self.b2 = nn.Parameter(torch.zeros(n_actions))

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        h = nn.functional.elu(torch.einsum("nd,kdh->nkh", phi, self.w1) + self.b1)
        return torch.einsum("nkh,kh->nk", h, self.w2) + self.b2


class FactoredHead(nn.Module):
    """One MLP over (phi, the call's features): shares strength across calls."""

    def __init__(self, d: int, action_features: np.ndarray, hidden: int):
        super().__init__()
        f = (action_features - action_features.mean(0)) / (action_features.std(0) + 1e-6)
        f = np.hstack([f, np.eye(len(action_features))])
        self.register_buffer("feats", torch.as_tensor(f, dtype=torch.float32))
        self.net = nn.Sequential(nn.Linear(d + f.shape[1], hidden), nn.ELU(), nn.Linear(hidden, 1))

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        n, k = phi.shape[0], self.feats.shape[0]
        z = torch.cat([phi.unsqueeze(1).expand(n, k, -1),
                       self.feats.unsqueeze(0).expand(n, k, -1)], dim=2)
        return self.net(z).squeeze(-1)


class TARNet(nn.Module):
    def __init__(self, n_numeric: int, cardinalities: list[int], n_actions: int,
                 hidden: list[int], head_hidden: int, dropout: float,
                 heads: str = "factored", action_features: np.ndarray | None = None,
                 embed_max: int = 8):
        super().__init__()
        self.phi = Representation(n_numeric, cardinalities, hidden, dropout, embed_max)
        d = self.phi.out_dim
        self.q_head = (FactoredHead(d, action_features, head_hidden) if heads == "factored"
                       else SeparateHeads(d, n_actions, head_hidden))

    def forward(self, num, cat) -> dict[str, torch.Tensor]:
        return {"q": self.q_head(self.phi(num, cat))}


class DragonNet(TARNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        n_actions = args[2]
        self.pi_head = nn.Linear(self.phi.out_dim, n_actions)
        self.epsilon = nn.Parameter(torch.zeros(n_actions))

    def forward(self, num, cat) -> dict[str, torch.Tensor]:
        z = self.phi(num, cat)
        return {"q": self.q_head(z), "logits": self.pi_head(z)}


def _loss(model: nn.Module, out: dict, a: torch.Tensor, y: torch.Tensor,
          alpha: float, beta: float, clip: float, outcome_only: bool = False) -> torch.Tensor:
    q_taken = out["q"].gather(1, a[:, None]).squeeze(1)
    loss = nn.functional.mse_loss(q_taken, y)
    if "logits" in out and not outcome_only:
        loss = loss + alpha * nn.functional.cross_entropy(out["logits"], a)
        pi = torch.softmax(out["logits"], dim=1).gather(1, a[:, None]).squeeze(1).clamp_min(clip)
        perturbed = q_taken + model.epsilon[a] / pi
        loss = loss + beta * nn.functional.mse_loss(perturbed, y)
    return loss


# ── Fitted model ─────────────────────────────────────────────────────────────
@dataclass
class NeuralQModel:
    net: nn.Module
    encoder: StateEncoder
    y_mean: float
    y_std: float
    n_actions: int
    state_cols: list[str]
    history: list[dict] = field(default_factory=list)
    q_scale: float = 1.0

    @torch.no_grad()
    def _forward(self, df: pd.DataFrame, key: str) -> np.ndarray:
        self.net.eval()
        num, cat = self.encoder.transform(df)
        out = [self.net(num[i:i + 8192], cat[i:i + 8192])[key] for i in range(0, len(df), 8192)]
        return torch.cat(out).numpy()

    def predict_all_actions(self, df: pd.DataFrame) -> np.ndarray:
        return self._forward(df, "q") * self.y_std + self.y_mean

    def predict_taken(self, df: pd.DataFrame, actions: np.ndarray) -> np.ndarray:
        return self.predict_all_actions(df)[np.arange(len(df)), np.asarray(actions)]

    def propensity(self, df: pd.DataFrame) -> np.ndarray:
        """DragonNet's own pi(a | s) (diagnostic; the pipeline's pi_b is separate)."""
        return torch.softmax(torch.as_tensor(self._forward(df, "logits")), dim=1).numpy()


class SnapBatches:
    """Shuffled minibatches sliced straight from in-memory tensors.

    The batches `DataLoader(TensorDataset(...), shuffle=True, generator=g)`
    would give — the same permutation stream, the same rows — without its
    per-row indexing and re-stacking, which dominated the epoch time (~10x) for
    networks this small.
    """

    def __init__(self, tensors: tuple[torch.Tensor, ...], batch_size: int,
                 generator: torch.Generator):
        self.tensors, self.batch_size, self.generator = tensors, batch_size, generator
        self.n = len(tensors[0])

    def __len__(self) -> int:
        return -(-self.n // self.batch_size)

    def __iter__(self):
        # Per epoch, DataLoader draws a worker base seed from the generator,
        # then RandomSampler a permutation — and a second, discarded one for the
        # (empty) remainder. Making the same draws keeps the stream identical.
        torch.empty((), dtype=torch.int64).random_(generator=self.generator)
        perm = torch.randperm(self.n, generator=self.generator)
        for i in range(0, self.n, self.batch_size):
            idx = perm[i:i + self.batch_size]
            yield tuple(t[idx] for t in self.tensors)
        torch.randperm(self.n, generator=self.generator)


def _game_split(df: pd.DataFrame, frac: float, seed: int) -> np.ndarray:
    """Validation mask by whole games — snaps of a game share a script."""
    games = df["game_id"].unique() if "game_id" in df.columns else np.arange(len(df))
    rng = np.random.default_rng(seed)
    val_games = set(rng.choice(games, max(1, int(frac * len(games))), replace=False))
    key = df["game_id"] if "game_id" in df.columns else pd.Series(np.arange(len(df)))
    return key.isin(val_games).to_numpy()


def fit_neural_q_model(ds: Dataset, cfg: dict) -> NeuralQModel:
    """Train the configured network (`outcome_model.type`: tarnet | dragonnet)
    with early stopping on held-out games."""
    ocfg, seed = cfg["outcome_model"], int(cfg["seed"])
    kind = ocfg["type"]
    if kind not in ARCHITECTURES:
        raise ValueError(f"unknown architecture '{kind}' ({' | '.join(ARCHITECTURES)})")
    ncfg = ocfg.get("neural", {})
    torch.manual_seed(seed)

    df = ds.df
    val = _game_split(df, ncfg.get("val_frac", 0.1), seed)
    enc = StateEncoder(ds.numeric_state, ds.categorical_state).fit(df[~val])
    y = df[ds.reward_col].to_numpy(float)
    y_mean, y_std = float(y[~val].mean()), float(y[~val].std())
    num, cat = enc.transform(df)
    a = torch.as_tensor(df[ds.action_col].to_numpy().copy(), dtype=torch.long)
    yt = torch.as_tensor((y - y_mean) / y_std, dtype=torch.float32)

    action_features = None
    heads = ncfg.get("heads", "factored")
    if heads not in ("factored", "separate"):
        raise ValueError(f"unknown neural.heads '{heads}' (factored | separate)")
    if heads == "factored":
        from ...data.features import action_feature_matrix
        action_features = action_feature_matrix(cfg)
    arch = DragonNet if kind == "dragonnet" else TARNet
    net = arch(num.shape[1], enc.cardinalities, ds.n_actions,
               ncfg.get("hidden", [128, 64]), ncfg.get("head_hidden", 32),
               ncfg.get("dropout", 0.1), heads, action_features, ncfg.get("embed_max", 8))
    opt = torch.optim.AdamW(net.parameters(), lr=ncfg.get("lr", 1e-3),
                            weight_decay=ncfg.get("weight_decay", 1e-4))
    alpha, beta = ncfg.get("alpha", 1.0), ncfg.get("beta", 1.0)
    clip = cfg["ope"]["clip_propensity"][0]
    # Early stopping on the training loss (default), or on the outcome error
    # alone, so the propensity and targeted terms do not decide when Q stops.
    early_stop = ncfg.get("early_stop", "loss")
    if early_stop not in ("loss", "outcome"):
        raise ValueError(f"unknown neural.early_stop '{early_stop}' (loss | outcome)")

    tr_idx = torch.as_tensor(np.flatnonzero(~val))
    va_idx = torch.as_tensor(np.flatnonzero(val))
    loader = SnapBatches((num[tr_idx], cat[tr_idx], a[tr_idx], yt[tr_idx]),
                         ncfg.get("batch_size", 1024), torch.Generator().manual_seed(seed))
    best, best_state, stale, history = np.inf, None, 0, []
    for epoch in range(ncfg.get("max_epochs", 60)):
        net.train()
        for xn, xc, aa, yy in loader:
            opt.zero_grad()
            _loss(net, net(xn, xc), aa, yy, alpha, beta, clip).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = float(_loss(net, net(num[va_idx], cat[va_idx]), a[va_idx], yt[va_idx],
                            alpha, beta, clip, outcome_only=early_stop == "outcome"))
        history.append({"epoch": epoch, "val_loss": v})
        if v < best - 1e-5:
            best, stale = v, 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            stale += 1
            if stale >= ncfg.get("patience", 5):
                break
    if best_state is None:
        last = history[-1]["val_loss"] if history else None
        raise ValueError(f"the {kind} never reached a finite validation loss "
                         f"(max_epochs={ncfg.get('max_epochs', 60)}, last val_loss={last}); "
                         "check the data for NaNs and the learning rate")
    net.load_state_dict(best_state)

    model = NeuralQModel(net=net, encoder=enc, y_mean=y_mean, y_std=y_std,
                         n_actions=ds.n_actions, state_cols=ds.state_cols, history=history)
    from .q_model import _within_state_spread
    model.q_scale = _within_state_spread(model.predict_all_actions(df))
    return model
