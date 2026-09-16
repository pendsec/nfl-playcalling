"""
Coverage-shell imputation — treating the latent treatment label as latent.

The shell is charted on ~94% of dropbacks and ~3% of runs. The defense still
CALLED a shell on those run snaps; NGS just records coverage as played, and a
run never develops one. So this fits P(shell | pre-snap state) on the charted
plays and samples from that posterior for the unlabeled ones.

Why SAMPLING and not argmax
---------------------------
Measured on 2022-24 (train: 2022-23 dropbacks; test: 2024):

    feature set            held-out dropbacks      charted runs
                           acc    logloss   H      acc    logloss   H
    marginal prior        0.411   1.566   1.620   0.435   1.536   1.620
    + FTN (box, motion)   0.366   1.502   1.439   0.363   1.512   1.430
    + team shell tendency 0.413   1.449   1.382   0.428   1.459   1.369

Accuracy never beats the marginal — taking the argmax would be no better than
writing the team's modal shell on every run, while *looking* like data. The
gain is entirely in the distribution (log loss 1.566 -> 1.449, entropy 1.620 ->
1.382). A calibrated posterior sampled m times carries that information and,
through `pooling.pool_rubin`, carries the uncertainty too.

Calibration (2022-23 fit, 2024 held out, temperature cross-fitted over 5 folds):

                        uncalibrated   calibrated
    confidence ECE            0.0321       0.0199
    classwise ECE             0.0378       0.0372
    Brier                     0.7180       0.7164

T lands near 1.07-1.13 — mild overconfidence. Note how little `classwise_ece`
moves: a single global temperature fixes the top-class confidence but barely
touches the per-class probabilities, and those are what sampling actually uses.

What this does NOT fix
----------------------
  * The outcome is deliberately EXCLUDED from the imputation features. Standard
    multiple-imputation practice says to include it, to avoid attenuating the
    X-Y association — but here the imputed variable IS the treatment whose
    effect on Y we then estimate, so conditioning on Y would let the imputer
    write in the very association the study is trying to measure. Excluding it
    attenuates the recovered effect toward null instead: a conservative bias,
    and the honest direction to err in.
  * The only run plays available to validate against are the ~6% NGS charted,
    and those are plays the defense defended AS passes (mean reward -0.39 vs
    +0.07 on uncharted runs, box 5.9 vs 6.8). Transfer measured on them is
    optimistic. `SELECTION_WARNING` records this next to the model.
  * Team shell tendency is computed from charted plays, i.e. dropbacks. Applying
    it to runs assumes a team's run-down shell mix resembles its dropback mix.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from ..calibration import TemperatureScaler, calibration_report
from .features import FORBIDDEN_FEATURES, imputer_features

SELECTION_WARNING = (
    "Validation against NGS-charted runs is optimistic: those are the runs the "
    "defense defended as passes (light box, in coverage), not a random sample "
    "of run snaps."
)


@dataclass
class ShellImputer:
    """A fitted posterior over coverage shells for unlabeled plays."""
    pipeline: Pipeline
    classes_: np.ndarray
    n_shells: int
    numeric_features: list[str]
    categorical_features: list[str]
    calibrator: TemperatureScaler = field(default_factory=TemperatureScaler)
    diagnostics: dict = field(default_factory=dict)

    @property
    def feature_cols(self) -> list[str]:
        return self.numeric_features + self.categorical_features

    def posterior(self, df: pd.DataFrame) -> np.ndarray:
        """P(shell = k | pre-snap state), shape (n, n_shells), rows summing to 1."""
        proba = self.pipeline.predict_proba(df[self.feature_cols])
        full = np.zeros((len(df), self.n_shells))
        full[:, self.classes_.astype(int)] = proba
        total = full.sum(axis=1, keepdims=True)
        full = np.divide(full, total, out=np.full_like(full, 1.0 / self.n_shells),
                         where=total > 0)
        # Temperature is monotone, so this changes the confidence, never the
        # ranking — accuracy is identical before and after.
        return self.calibrator.transform(full)

    def draw(self, df: pd.DataFrame, n_draws: int, seed: int = 0) -> np.ndarray:
        """`n_draws` independent samples from the posterior, shape (n_draws, n).

        Weighted sampling, not argmax: each play is drawn from its own posterior
        every time, so the spread ACROSS draws is what `pool_rubin` converts into
        the interval widening that label uncertainty demands.
        """
        p = self.posterior(df)
        rng = np.random.default_rng(seed)
        # Inverse-CDF sampling, vectorized over plays: one uniform per play per
        # draw, compared against each row's cumulative posterior.
        cdf = np.cumsum(p, axis=1)
        cdf[:, -1] = 1.0
        u = rng.random((n_draws, len(p), 1))
        return (u > cdf[None, :, :]).sum(axis=2).clip(0, self.n_shells - 1)

    def information_ratio(self, df: pd.DataFrame) -> float:
        """1 - H(posterior)/H(uniform): share of shell uncertainty removed."""
        p = self.posterior(df)
        h = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum(axis=1).mean()
        return float(1.0 - h / np.log(self.n_shells))


def _assert_presnap(features: list[str]) -> None:
    bad = sorted(set(features) & FORBIDDEN_FEATURES)
    if bad:
        raise ValueError(
            f"Post-snap or outcome column(s) in the imputation features: {bad}. "
            "The imputed variable is the treatment; conditioning it on the "
            "outcome would write the effect being measured into the data."
        )


def fit_shell_imputer(
    labeled: pd.DataFrame,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    shell_col: str = "shell",
    n_shells: int = 6,
    seed: int = 42,
    calibration_folds: int = 5,
    holdout_season: int | None = None,
) -> ShellImputer:
    """Fit a temperature-calibrated shell posterior on plays whose shell IS charted.

    `labeled` must contain only charted plays. Features default to the repo's
    declared set (`features.imputer_features`) so that every caller trains the
    same model; pass them explicitly only to run an ablation.

    The temperature is CROSS-FITTED over `calibration_folds` (see
    `_fit_with_temperature` for why a plain holdout is wrong here). When
    `holdout_season` is given, diagnostics are computed on that season with a
    model — and a temperature — that never saw it. Otherwise diagnostics are
    omitted rather than reported in-sample.
    """
    if numeric_features is None or categorical_features is None:
        declared_num, declared_cat = imputer_features(n_shells)
        numeric_features = declared_num if numeric_features is None else numeric_features
        categorical_features = (declared_cat if categorical_features is None
                                else categorical_features)
    features = numeric_features + categorical_features
    _assert_presnap(features)

    def _build() -> Pipeline:
        prep = ColumnTransformer([
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             categorical_features),
        ])
        base = LGBMClassifier(n_estimators=400, learning_rate=0.05, max_depth=6,
                              random_state=seed, verbose=-1)
        return Pipeline([("prep", prep), ("clf", base)])

    def _raw(pipe: Pipeline, frame: pd.DataFrame) -> np.ndarray:
        proba = pipe.predict_proba(frame[features])
        full = np.zeros((len(frame), n_shells))
        full[:, pipe.named_steps["clf"].classes_.astype(int)] = proba
        total = full.sum(axis=1, keepdims=True)
        return np.divide(full, total, out=np.full_like(full, 1.0 / n_shells),
                         where=total > 0)

    def _fit_with_temperature(frame: pd.DataFrame) -> tuple[Pipeline, TemperatureScaler]:
        """Fit the base model on `frame`, with a CROSS-FITTED temperature.

        Temperature must be calibrated for the model it is applied to. A plain
        holdout gets this wrong: fitting T against a probe trained on half the
        rows, then applying it to a model trained on all of them, over-corrects
        badly — the data-starved probe is far more overconfident, so its T
        (1.88 on real data) drove the full model's confidence ECE from 0.032 to
        0.096. Calibration made calibration worse.

        Out-of-fold probabilities fix that: every row is scored by a model
        trained on (k-1)/k of the data, so T is fitted for a model close to the
        final one. Folds are capped by the rarest shell so the stratified split
        stays possible.
        """
        y_frame = frame[shell_col].astype(int).to_numpy()
        scaler = TemperatureScaler()
        folds = min(calibration_folds, int(pd.Series(y_frame).value_counts().min()))
        if folds >= 2 and len(frame) >= 200:
            oof = np.zeros((len(frame), n_shells))
            skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
            for tr_idx, te_idx in skf.split(frame, y_frame):
                fold_pipe = _build().fit(frame.iloc[tr_idx][features], y_frame[tr_idx])
                oof[te_idx] = _raw(fold_pipe, frame.iloc[te_idx])
            scaler = TemperatureScaler.fit(oof, y_frame)
        return _build().fit(frame[features], y_frame), scaler

    diagnostics: dict = {"selection_warning": SELECTION_WARNING,
                         "numeric_features": list(numeric_features),
                         "categorical_features": list(categorical_features)}

    if holdout_season is not None and "season" in labeled.columns:
        tr = labeled[labeled["season"] < holdout_season]
        te = labeled[labeled["season"] >= holdout_season]
        if len(tr) > 50 and len(te) > 50:
            probe_pipe, probe_scaler = _fit_with_temperature(tr)
            raw = _raw(probe_pipe, te)
            cal = probe_scaler.transform(raw)
            y_te = te[shell_col].to_numpy().astype(int)
            prior = np.bincount(tr[shell_col].astype(int), minlength=n_shells) / len(tr)
            h = -(cal * np.log(np.clip(cal, 1e-12, 1.0))).sum(axis=1).mean()
            diagnostics.update(
                n_train=len(tr), n_holdout=len(te), holdout_season=holdout_season,
                diagnostic_temperature=probe_scaler.temperature,
                accuracy=float(accuracy_score(y_te, cal.argmax(axis=1))),
                marginal_accuracy=float((y_te == prior.argmax()).mean()),
                log_loss=float(log_loss(y_te, cal, labels=list(range(n_shells)))),
                marginal_log_loss=float(log_loss(
                    y_te, np.tile(prior, (len(y_te), 1)), labels=list(range(n_shells)))),
                information_ratio=float(1.0 - h / np.log(n_shells)),
                **calibration_report(y_te, cal),
                **{f"uncalibrated_{k}": v for k, v in calibration_report(y_te, raw).items()},
            )

    # Final model: all labeled data, with its own cross-fitted temperature.
    pipe, scaler = _fit_with_temperature(labeled)
    diagnostics["temperature"] = scaler.temperature
    diagnostics["calibration_folds"] = calibration_folds
    return ShellImputer(pipe, pipe.named_steps["clf"].classes_, n_shells,
                        numeric_features, categorical_features, scaler, diagnostics)
