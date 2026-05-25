"""
Q13 – RL Algorithm: Linear Q-learning with Tile Coding
========================================================
Algorithm choice: Tabular / Linear Q-learning with tile coding (no deep learning).

Why Q-learning with tile coding?
  • The state space is continuous (15 PCA dims), so raw tabular Q-tables are
    infeasible. Tile coding discretises the continuous state into sparse binary
    features, enabling linear function approximation of Q-values without neural nets.
  • Q-learning is off-policy (learns the greedy policy from exploratory data),
    which is well-suited to batch-style claim processing.
  • Simple, interpretable, and provably convergent under standard conditions.
  • Epsilon-greedy exploration gradually reduces to near-greedy exploitation.

Dependencies: numpy, matplotlib  (no torch / tensorflow)
Import:  from q12_rl_environment import build_claims_env, ClaimsProcessingEnv
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import deque


# ── tile coding ────────────────────────────────────────────────────────────────

class TileCoder:
    """
    Tile coding for continuous state spaces.
    Maps a state vector to a set of active tile indices (sparse binary features).

    Parameters
    ----------
    n_tilings   : number of overlapping grids (8 is standard)
    n_tiles     : tiles per dimension per tiling
    state_low   : min value per dimension
    state_high  : max value per dimension
    n_actions   : number of actions (for separate weight vector)
    """

    def __init__(self, n_tilings: int, n_tiles: int,
                 state_low: np.ndarray, state_high: np.ndarray,
                 n_actions: int):
        self.n_tilings = n_tilings
        self.n_tiles = n_tiles
        self.n_actions = n_actions
        self.state_dim = len(state_low)

        # offsets shift each tiling so they don't perfectly overlap
        self.offsets = np.array([
            (np.arange(self.state_dim) + tiling * 0.5) / n_tilings
            for tiling in range(n_tilings)
        ])  # (n_tilings, state_dim)

        self.low  = state_low
        self.scale = (state_high - state_low + 1e-8)

        # total features = n_tilings * n_tiles^state_dim …  but we hash for
        # memory efficiency using a hash table of size hash_size
        self.hash_size = n_tilings * (n_tiles ** 2) * 4   # heuristic
        # weights: one weight per (action, hash_bucket)
        self.weights = np.zeros((n_actions, self.hash_size))

    def _tile_indices(self, state: np.ndarray) -> np.ndarray:
        """Return active tile indices for this state (length = n_tilings)."""
        # normalise to [0, n_tiles)
        norm = np.clip((state - self.low) / self.scale, 0, 1 - 1e-8) * self.n_tiles
        indices = []
        for t in range(self.n_tilings):
            # shift and floor
            floored = (norm + self.offsets[t] * self.n_tiles).astype(int)
            # hash the multi-dim tile coordinate + tiling index
            h = hash(tuple(floored) + (t,)) % self.hash_size
            indices.append(h)
        return np.array(indices)

    def q_values(self, state: np.ndarray) -> np.ndarray:
        """Compute Q(s, a) for all actions."""
        indices = self._tile_indices(state)
        return self.weights[:, indices].sum(axis=1)   # (n_actions,)

    def update(self, state: np.ndarray, action: int, target: float, alpha: float):
        """Semi-gradient update for one (s, a, target) triplet."""
        indices = self._tile_indices(state)
        q_sa = self.weights[action, indices].sum()
        error = target - q_sa
        # divide by n_tilings to keep step size consistent
        self.weights[action, indices] += alpha * error / self.n_tilings


# ── Q-learning agent ───────────────────────────────────────────────────────────

class QLearningAgent:
    """
    Off-policy Q-learning with ε-greedy exploration and tile-coded
    linear function approximation.

    Hyperparameters
    ---------------
    alpha       : learning rate (step size)
    gamma       : discount factor (1.0 for episodic tasks is fine here;
                  each claim is independent so discounting is less critical)
    eps_start   : initial exploration rate
    eps_end     : final exploration rate
    eps_decay   : multiplicative decay per episode
    n_tilings   : tile coding parameter
    n_tiles     : tile coding parameter
    """

    def __init__(self,
                 state_low: np.ndarray,
                 state_high: np.ndarray,
                 n_actions: int = 4,
                 alpha: float = 0.05,
                 gamma: float = 0.95,
                 eps_start: float = 1.0,
                 eps_end: float = 0.05,
                 eps_decay: float = 0.97,
                 n_tilings: int = 8,
                 n_tiles: int = 6,
                 random_state: int = 42):

        self.n_actions = n_actions
        self.alpha     = alpha
        self.gamma     = gamma
        self.eps       = eps_start
        self.eps_end   = eps_end
        self.eps_decay = eps_decay
        self.rng       = np.random.default_rng(random_state)

        self.tc = TileCoder(n_tilings, n_tiles, state_low, state_high, n_actions)

        # tracking
        self.episode_rewards  = []
        self.episode_avg_q    = []
        self.episode_epsilons = []

    # -- policy ----------------------------------------------------------------

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        if (not greedy) and (self.rng.random() < self.eps):
            return int(self.rng.integers(0, self.n_actions))
        q = self.tc.q_values(state)
        return int(np.argmax(q))

    # -- training --------------------------------------------------------------

    def train_episode(self, env) -> float:
        """Run one episode (full pass over train set). Returns total reward."""
        state = env.reset()
        total_reward = 0.0
        q_sum = 0.0
        steps = 0

        while not env.done:
            action = self.select_action(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            # Q-learning target
            if done or next_state is None:
                target = reward
            else:
                target = reward + self.gamma * np.max(self.tc.q_values(next_state))

            self.tc.update(state, action, target, self.alpha)

            q_sum += np.max(self.tc.q_values(state))
            steps += 1
            state = next_state if next_state is not None else state

        self.episode_rewards.append(total_reward)
        self.episode_avg_q.append(q_sum / max(steps, 1))
        self.episode_epsilons.append(self.eps)

        # decay epsilon
        self.eps = max(self.eps_end, self.eps * self.eps_decay)
        return total_reward

    def evaluate_episode(self, env) -> tuple:
        """
        Greedy evaluation pass. Returns (total_reward, action_counts).
        """
        state = env.reset()
        total_reward = 0.0
        action_counts = np.zeros(self.n_actions, dtype=int)

        while not env.done:
            action = self.select_action(state, greedy=True)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            action_counts[action] += 1
            state = next_state if next_state is not None else state

        return total_reward, action_counts


# ── training loop ──────────────────────────────────────────────────────────────

def train_agent(train_env,
                test_env,
                n_episodes: int = 60,
                eval_every: int = 5,
                alpha: float = 0.05,
                gamma: float = 0.95,
                eps_start: float = 1.0,
                eps_end: float = 0.05,
                eps_decay: float = 0.97,
                n_tilings: int = 8,
                n_tiles: int = 6,
                random_state: int = 42):
    """
    Full training loop.

    Returns
    -------
    agent, eval_rewards (list), eval_episodes (list)
    """

    # estimate state bounds from training data for tile coder
    state_low  = train_env.X.min(axis=0)
    state_high = train_env.X.max(axis=0)

    agent = QLearningAgent(
        state_low=state_low, state_high=state_high,
        n_actions=train_env.n_actions,
        alpha=alpha, gamma=gamma,
        eps_start=eps_start, eps_end=eps_end, eps_decay=eps_decay,
        n_tilings=n_tilings, n_tiles=n_tiles,
        random_state=random_state
    )

    eval_rewards  = []
    eval_episodes = []

    print(f"{'Episode':>8}  {'Train Reward':>14}  {'Test Reward':>12}  {'Epsilon':>8}")
    print("-" * 52)

    for ep in range(1, n_episodes + 1):
        train_reward = agent.train_episode(train_env)

        if ep % eval_every == 0 or ep == 1:
            test_reward, _ = agent.evaluate_episode(test_env)
            eval_rewards.append(test_reward)
            eval_episodes.append(ep)
            print(f"{ep:>8}  {train_reward:>14.1f}  {test_reward:>12.1f}  {agent.eps:>8.4f}")

    print("\n[Q13] Training complete.")
    return agent, eval_rewards, eval_episodes


# ── learning curve plot ────────────────────────────────────────────────────────

def plot_learning_curve(agent, eval_rewards, eval_episodes, smooth_window: int = 5):
    """
    Plot the learning curve:
      - Smoothed training rewards per episode
      - Test rewards at evaluation checkpoints
      - Epsilon decay over time
      - Average max-Q per episode
    """

    def smooth(x, w):
        if len(x) < w:
            return x
        return np.convolve(x, np.ones(w) / w, mode="valid")

    eps_arr     = agent.episode_epsilons
    train_r_arr = agent.episode_rewards
    avg_q_arr   = agent.episode_avg_q
    episodes    = np.arange(1, len(train_r_arr) + 1)

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── (0,0) train + test rewards ──────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    smoothed = smooth(train_r_arr, smooth_window)
    ep_smooth = episodes[smooth_window - 1:]
    ax0.plot(episodes, train_r_arr, color="#bdc3c7", lw=0.8, label="Train (raw)")
    ax0.plot(ep_smooth, smoothed,   color="#2980b9", lw=2,   label=f"Train (smooth {smooth_window})")
    ax0.plot(eval_episodes, eval_rewards, "o-", color="#e74c3c", lw=2, ms=5, label="Test (greedy)")
    ax0.set_title("Learning Curve: Episode Rewards", fontweight="bold")
    ax0.set_xlabel("Episode"); ax0.set_ylabel("Total Reward")
    ax0.legend(fontsize=8); ax0.grid(alpha=0.3)

    # ── (0,1) epsilon decay ─────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(episodes, eps_arr, color="#8e44ad", lw=2)
    ax1.fill_between(episodes, 0, eps_arr, alpha=0.15, color="#8e44ad")
    ax1.set_title("Epsilon Decay (Exploration → Exploitation)", fontweight="bold")
    ax1.set_xlabel("Episode"); ax1.set_ylabel("ε (epsilon)")
    ax1.grid(alpha=0.3)

    # ── (1,0) avg max-Q ─────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    q_smooth = smooth(avg_q_arr, smooth_window)
    ax2.plot(episodes, avg_q_arr, color="#bdc3c7", lw=0.8, label="Raw")
    ax2.plot(episodes[smooth_window - 1:], q_smooth, color="#27ae60", lw=2, label=f"Smooth {smooth_window}")
    ax2.set_title("Average Max Q-value per Episode", fontweight="bold")
    ax2.set_xlabel("Episode"); ax2.set_ylabel("Avg max Q(s,a)")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    # ── (1,1) per-episode reward histogram ──────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(train_r_arr[:len(train_r_arr)//2], bins=20, alpha=0.6,
             color="#e67e22", label="First half episodes")
    ax3.hist(train_r_arr[len(train_r_arr)//2:], bins=20, alpha=0.6,
             color="#2980b9", label="Second half episodes")
    ax3.set_title("Reward Distribution: Early vs Late Training", fontweight="bold")
    ax3.set_xlabel("Total Episode Reward"); ax3.set_ylabel("Count")
    ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

    fig.suptitle("Q13 – Q-Learning with Tile Coding: Learning Curves",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.savefig("q13_learning_curves.png", bbox_inches="tight", dpi=150)
    plt.show()
    print("[Q13] Learning curve saved to q13_learning_curves.png")


# ── agent behaviour explanation ────────────────────────────────────────────────

def explain_agent_behaviour(agent, test_env):
    """
    Print a concise analysis of what the trained agent has learned:
    action distribution, Q-value spread, and per-action reward contribution.
    """
    state = test_env.reset()
    action_counts   = np.zeros(agent.n_actions, dtype=int)
    label_action    = {0: np.zeros(agent.n_actions), 1: np.zeros(agent.n_actions)}
    total_reward    = 0.0
    q_by_action     = [[] for _ in range(agent.n_actions)]

    while not test_env.done:
        q_vals = agent.tc.q_values(state)
        action = int(np.argmax(q_vals))
        next_state, reward, done, info = test_env.step(action)

        action_counts[action] += 1
        label_action[info["true_label"]][action] += 1
        total_reward += reward
        q_by_action[action].append(q_vals[action])

        state = next_state if next_state is not None else state

    n = action_counts.sum()
    action_names = test_env.action_names

    print("\n" + "=" * 60)
    print("Q13 – Agent Behaviour Analysis (greedy on test set)")
    print("=" * 60)
    print(f"Total reward: {total_reward:,.1f}   Claims: {n}\n")
    print(f"{'Action':<20} {'Count':>7} {'%':>7} {'Avg Q':>8}")
    print("-" * 45)
    for a in range(agent.n_actions):
        avg_q = np.mean(q_by_action[a]) if q_by_action[a] else 0.0
        print(f"{action_names[a]:<20} {action_counts[a]:>7} {100*action_counts[a]/n:>6.1f}%  {avg_q:>8.3f}")

    print("\nAction distribution BY TRUE LABEL")
    print(f"{'Action':<20} {'Benign (1)':>12} {'Risky (0)':>12}")
    print("-" * 45)
    for a in range(agent.n_actions):
        print(f"{action_names[a]:<20} {int(label_action[1][a]):>12} {int(label_action[0][a]):>12}")
    print("=" * 60)

    return action_counts, label_action


# ── main (end-to-end example) ──────────────────────────────────────────────────

if __name__ == "__main__":
    # Requires q12_rl_environment.py to be importable
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from q12_rl_environment import build_claims_env

    CSV_PATH = "test_data.csv"   # <- update path as needed
    train_env, test_env, pca, scaler, X_te, y_te = build_claims_env(CSV_PATH)

    agent, eval_rewards, eval_episodes = train_agent(
        train_env, test_env,
        n_episodes=60,
        eval_every=5,
    )

    plot_learning_curve(agent, eval_rewards, eval_episodes)
    explain_agent_behaviour(agent, test_env)