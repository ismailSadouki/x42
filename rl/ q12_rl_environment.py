"""
Q12 - Reinforcement Learning Environment for Claims Processing
================================================================
Implements a custom Gym-like environment for claims adjudication.

State  : PCA-reduced features (top 15 PCs, matching Part I findings that
         15 components satisfy the Kaiser criterion and capture ~93 % variance)
Actions: 0=Approve (fast-track)  1=Request docs  2=Manual review  3=Deny
Reward : Designed to balance correctness, fraud risk, delays and satisfaction.

Run this file standalone OR import build_claims_env() in your notebook.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# ── helpers ────────────────────────────────────────────────────────────────────

def load_and_preprocess(csv_path: str, sample_frac: float = 0.40, random_state: int = 42):
    """
    Load test_data.csv, draw a stratified sample (~40 % as exam recommends),
    impute, encode, scale, and return X_scaled and y.
    """
    df = pd.read_csv(csv_path)

    # stratified sample to keep class ratio
    df = df.groupby("target", group_keys=False).apply(
        lambda g: g.sample(frac=sample_frac, random_state=random_state)
    ).reset_index(drop=True)

    y = df["target"].values
    drop_cols = ["target"]
    if "ID" in df.columns:
        drop_cols.append("ID")

    X = df.drop(columns=drop_cols)

    # encode categoricals
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        X[col] = pd.Categorical(X[col]).codes.astype(float)

    # median-impute
    X = X.fillna(X.median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler


def apply_pca(X_scaled, n_components: int = 15, random_state: int = 42):
    """
    Reduce to n_components PCs (Part I recommends 15 for Kaiser criterion).
    Returns transformed array and fitted PCA object.
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_.cumsum()[-1]
    print(f"[PCA] {n_components} components → {explained*100:.1f}% variance explained")
    return X_pca, pca


# ── environment ────────────────────────────────────────────────────────────────

class ClaimsProcessingEnv:
    """
    Gym-compatible environment for claims adjudication.

    Justification of design choices
    --------------------------------
    State  (15 PCA dims)
      • Part I showed 4 PCs cover 80 % variance; Kaiser criterion selects 15.
        Using 15 gives the agent richer signal while staying compact. Raw 131
        features would make Q-table or tile-coding infeasible.
      • Features are already standardised, so the action-value surface is
        smooth and well-conditioned for function approximation.

    Action space (discrete, 4 actions)
      0 – Approve immediately (fast-track)
      1 – Request additional documentation
      2 – Refer to manual review
      3 – Deny claim

    Reward function
      The reward is claim-specific and depends on the TRUE label (target):
        target=1 → claim is benign / suitable for fast-track
        target=0 → claim is risky / requires standard processing

      | Action          | target=1 (benign) | target=0 (risky) |
      |-----------------|-------------------|------------------|
      | Approve         | +10               | -15  (fraud risk!)|
      | Request docs    |  -1               |  +3  (cautious)  |
      | Manual review   |  -2               |  +5  (correct)   |
      | Deny            |  -5               |  +2  (avoided)   |

      Rationale:
      • Approving a risky claim is the worst error (-15): large negative as
        specified in the exam ("large negative reward for incorrect fast-tracking").
      • Correct approval of benign claim is the best outcome (+10): business
        value from fast-tracking.
      • Requesting docs always incurs a small delay cost (-1 for benign) but
        is rewarded for risky claims (+3).
      • Manual review is correct for risky claims (+5) but wastes resources
        on benign ones (-2).
      • Denial hurts customer satisfaction on benign claims (-5) but avoids
        payout on risky ones (+2).
    """

    # reward matrix: REWARD[action][target]
    REWARD = {
        0: {1: +10, 0: -15},   # Approve
        1: {1:  -1, 0:  +3},   # Request docs
        2: {1:  -2, 0:  +5},   # Manual review
        3: {1:  -5, 0:  +2},   # Deny
    }

    ACTION_NAMES = {
        0: "Approve",
        1: "Request docs",
        2: "Manual review",
        3: "Deny",
    }

    def __init__(self, X_pca: np.ndarray, y: np.ndarray, shuffle: bool = True,
                 random_state: int = 42):
        """
        Parameters
        ----------
        X_pca   : (n_samples, n_components) PCA-reduced feature matrix
        y       : (n_samples,) binary target  (1=benign, 0=risky)
        shuffle : randomise episode order each reset
        """
        self.X = X_pca.astype(np.float32)
        self.y = y.astype(int)
        self.n_samples = len(y)
        self.n_actions = 4
        self.state_dim = X_pca.shape[1]
        self.shuffle = shuffle
        self.rng = np.random.default_rng(random_state)

        self._order = np.arange(self.n_samples)
        self._idx = 0
        self.done = True

    # -- gym interface ----------------------------------------------------------

    def reset(self):
        """Start a new episode (one pass over the dataset)."""
        if self.shuffle:
            self.rng.shuffle(self._order)
        self._idx = 0
        self.done = False
        return self._current_state()

    def step(self, action: int):
        """
        Apply action to current claim.
        Returns (next_state, reward, done, info).
        """
        assert not self.done, "Call reset() before step()"
        assert action in range(self.n_actions), f"Invalid action {action}"

        true_label = self.y[self._order[self._idx]]
        reward = self.REWARD[action][true_label]

        info = {
            "claim_idx": self._order[self._idx],
            "true_label": true_label,
            "action_name": self.ACTION_NAMES[action],
            "reward": reward,
        }

        self._idx += 1
        self.done = self._idx >= self.n_samples

        next_state = self._current_state() if not self.done else None
        return next_state, reward, self.done, info

    def _current_state(self):
        return self.X[self._order[self._idx]].copy()

    @property
    def action_names(self):
        return list(self.ACTION_NAMES.values())


# ── factory ────────────────────────────────────────────────────────────────────

def build_claims_env(csv_path: str,
                     n_components: int = 15,
                     sample_frac: float = 0.40,
                     test_size: float = 0.20,
                     random_state: int = 42):
    """
    Full pipeline: load → preprocess → PCA → split → build train/test envs.

    Returns
    -------
    train_env, test_env, pca, scaler, X_test_pca, y_test
    """
    print("[Q12] Loading and preprocessing data …")
    X_scaled, y, scaler = load_and_preprocess(csv_path, sample_frac, random_state)

    print("[Q12] Applying PCA …")
    X_pca, pca = apply_pca(X_scaled, n_components, random_state)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_pca, y, test_size=test_size, stratify=y, random_state=random_state
    )

    train_env = ClaimsProcessingEnv(X_tr, y_tr, shuffle=True, random_state=random_state)
    test_env  = ClaimsProcessingEnv(X_te, y_te, shuffle=False)

    print(f"[Q12] Train env: {len(y_tr)} claims | Test env: {len(y_te)} claims")
    print(f"[Q12] State dim: {n_components} | Actions: {train_env.action_names}")
    print("[Q12] Environment ready.\n")

    return train_env, test_env, pca, scaler, X_te, y_te


# ── quick self-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Smoke test with synthetic data (no CSV needed)
    rng = np.random.default_rng(0)
    X_fake = rng.standard_normal((500, 15)).astype(np.float32)
    y_fake = rng.integers(0, 2, size=500)

    env = ClaimsProcessingEnv(X_fake, y_fake)
    state = env.reset()
    total_r = 0
    while not env.done:
        action = rng.integers(0, 4)
        next_state, reward, done, info = env.step(int(action))
        total_r += reward

    print(f"Smoke test passed. Random agent total reward: {total_r}")
    print("Reward table:")
    for a, name in ClaimsProcessingEnv.ACTION_NAMES.items():
        print(f"  {name:20s}: benign={ClaimsProcessingEnv.REWARD[a][1]:+3d}  risky={ClaimsProcessingEnv.REWARD[a][0]:+3d}")