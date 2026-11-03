"""
Reinforcement Learning on RANDOMLY GENERATED mazes
===================================================
Needs maze_rl.py in the same folder (re-uses MazeEnv, TabularAgent, moving_avg).

Random maze generator
    10x10 grid, S at top-left, G at bottom-right, random walls and traps.
    Every maze is checked with BFS: a SAFE path (no traps) must exist, so every
    maze is solvable. BFS also gives the optimal path length, used as a yardstick.

Two experiments
  Experiment A - "Learn each maze"  (position state, fresh Q-table per maze)
      Shows the Q-learning agent works on ANY random maze, not just one layout.
      State = (row, col).  Nothing is transferred between mazes.

  Experiment B - "Generalise to unseen mazes"  (local-observation state, ONE Q-table)
      The agent is trained on a stream of different random mazes, then tested on
      100 brand-new mazes with learning switched off.
      State = (what is in the 4 neighbouring cells: free/wall/trap,
               direction of goal: sign of drow, sign of dcol,
               which of the 4 neighbours were already visited this episode)
      Because the state describes the surroundings instead of coordinates,
      what is learnt transfers to mazes never seen before.

Usage
    python maze_rl_random.py                 # both experiments
    python maze_rl_random.py --only A        # or --only B
    python maze_rl_random.py --mazes 20 --train-mazes 40000 --seed 1
"""
import argparse
import os
from collections import deque

import numpy as np

from maze_rl import MazeEnv, TabularAgent, moving_avg

N = 10


# ----------------------------------------------------------------------------
# 1. RANDOM MAZE GENERATOR
# ----------------------------------------------------------------------------
def bfs_shortest(grid):
    """Length of shortest path S->G that avoids walls AND traps (None if none)."""
    n, m = len(grid), len(grid[0])
    start, goal = (0, 0), (n - 1, m - 1)
    q, seen = deque([(start, 0)]), {start}
    while q:
        (r, c), d = q.popleft()
        if (r, c) == goal:
            return d
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < m and (nr, nc) not in seen and grid[nr][nc] in ".G":
                seen.add((nr, nc))
                q.append(((nr, nc), d + 1))
    return None


def random_layout(rng, size=N, wall_p=0.30, trap_p=0.04):
    """Return (layout, optimal_steps). Re-sample until a safe path exists."""
    while True:
        g = [["."] * size for _ in range(size)]
        for r in range(size):
            for c in range(size):
                x = rng.random()
                if x < wall_p:
                    g[r][c] = "#"
                elif x < wall_p + trap_p:
                    g[r][c] = "T"
        g[0][0], g[size - 1][size - 1] = "S", "G"
        opt = bfs_shortest([[("." if ch == "S" else ch) for ch in row] for row in g])
        if opt is not None:
            return ["".join(row) for row in g], opt


# ----------------------------------------------------------------------------
# 2. LOCAL-OBSERVATION ENVIRONMENT (for generalisation)
# ----------------------------------------------------------------------------
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


class LocalObsMaze(MazeEnv):
    """Same maze/rewards as MazeEnv, but the state is a local observation."""
    N_OBS = (3 ** 4) * 9 * 16

    def __init__(self, layout, max_steps=200):
        super().__init__(layout, max_steps)
        self.n_states = self.N_OBS

    def reset(self):
        super().reset()
        self.visited = {self.pos}
        return self._obs()

    def _cell(self, r, c):
        if not (0 <= r < self.n_rows and 0 <= c < self.n_cols) or self.grid[r][c] == "#":
            return 1
        return 2 if self.grid[r][c] == "T" else 0

    def _obs(self):
        r, c = self.pos
        idx = 0
        for dr, dc in DIRS:
            idx = idx * 3 + self._cell(r + dr, c + dc)
        gr = int(np.sign(self.goal[0] - r)) + 1
        gc = int(np.sign(self.goal[1] - c)) + 1
        idx = idx * 9 + gr * 3 + gc
        vis = 0
        for dr, dc in DIRS:
            vis = vis * 2 + ((r + dr, c + dc) in self.visited)
        return idx * 16 + vis

    def step(self, action):
        _, r, done, info = super().step(action)
        self.visited.add(self.pos)
        return self._obs(), r, done, info


# ----------------------------------------------------------------------------
# 3. EXPERIMENT A - fresh agent per random maze
# ----------------------------------------------------------------------------
def run_episode_learning(env, agent):
    s, total, done = env.reset(), 0.0, False
    a = agent.act(s)
    while not done:
        s2, r, done, info = env.step(a)
        a2 = agent.act(s2)
        agent.update(s, a, r, s2, done, a2)
        s, a, total = s2, a2, total + r
    return total, env.steps, info["success"]


def greedy_run(env, agent):
    s, done = env.reset(), False
    while not done:
        s, _, done, info = env.step(agent.act(s, greedy=True))
    return info["success"], env.steps


def experiment_A(n_mazes, episodes, seed):
    print(f"\n=== EXPERIMENT A: fresh Q-learning agent on each of {n_mazes} random mazes ===")
    rng = np.random.default_rng(seed)
    curves_r, curves_s, rows = [], [], []
    for i in range(n_mazes):
        layout, opt = random_layout(rng)
        env = MazeEnv(layout)
        agent = TabularAgent(env.n_states, 4, "qlearning", rng=np.random.default_rng(seed + i))
        rew, suc = [], []
        for _ in range(episodes):
            tot, _, ok = run_episode_learning(env, agent)
            agent.decay()
            rew.append(tot); suc.append(1 if ok else 0)
        ok, steps = greedy_run(env, agent)
        ratio = steps / opt if ok else np.nan
        rows.append((ok, steps, opt, ratio))
        curves_r.append(rew); curves_s.append(suc)
        print(f"  maze {i+1:2d}: optimal={opt:2d} | learnt path={steps:3d} | "
              f"reached goal={ok} | steps/optimal={ratio:.2f}")
    ok = np.array([r[0] for r in rows])
    ratios = np.array([r[3] for r in rows], float)
    print(f"\n  Solved {ok.sum()}/{n_mazes} mazes ({100*ok.mean():.0f}%) | "
          f"mean steps/optimal = {np.nanmean(ratios):.2f} (1.00 = shortest path)")
    return np.array(curves_r), np.array(curves_s), rows


# ----------------------------------------------------------------------------
# 4. EXPERIMENT B - one agent, many mazes, test on unseen ones
# ----------------------------------------------------------------------------
def evaluate_unseen(agent, n_test, seed, size=N, random_policy=False):
    rng = np.random.default_rng(seed)
    pol_rng = np.random.default_rng(seed + 1)
    succ, ratios, rews = [], [], []
    for _ in range(n_test):
        layout, opt = random_layout(rng, size)
        env = LocalObsMaze(layout)
        s, done, tot = env.reset(), False, 0.0
        while not done:
            a = int(pol_rng.integers(4)) if random_policy else agent.act(s, greedy=True)
            s, r, done, info = env.step(a)
            tot += r
        succ.append(info["success"]); rews.append(tot)
        if info["success"]:
            ratios.append(env.steps / opt)
    return dict(success=100 * np.mean(succ),
                ratio=float(np.mean(ratios)) if ratios else float("nan"),
                reward=float(np.mean(rews)))


def experiment_B(train_mazes, seed, n_test=100):
    print(f"\n=== EXPERIMENT B: ONE agent trained on {train_mazes} different random mazes ===")
    rng = np.random.default_rng(seed + 1000)
    agent = TabularAgent(LocalObsMaze.N_OBS, 4, "qlearning", alpha=0.03, gamma=0.97,
                         eps_start=1.0, eps_end=0.05,
                         eps_decay=0.05 ** (1.0 / (0.7 * train_mazes)),   # reaches floor at 70%
                         rng=np.random.default_rng(seed))
    log = {"reward": [], "success": [], "steps": []}
    checkpoints = []
    every = max(1, train_mazes // 10)
    base = evaluate_unseen(None, n_test, seed + 5000, random_policy=True)
    for ep in range(train_mazes):
        layout, _ = random_layout(rng)
        env = LocalObsMaze(layout)
        tot, steps, ok = run_episode_learning(env, agent)
        agent.decay()
        log["reward"].append(tot); log["steps"].append(steps); log["success"].append(1 if ok else 0)
        if (ep + 1) % every == 0:
            m = evaluate_unseen(agent, 100, seed + 7000)
            checkpoints.append((ep + 1, m["success"]))
            print(f"  maze #{ep+1:6d} | train success(last 500) "
                  f"{100*np.mean(log['success'][-500:]):5.1f}% | UNSEEN-maze success {m['success']:5.1f}% "
                  f"| eps {agent.eps:.2f}")
    final = evaluate_unseen(agent, n_test, seed + 5000)
    print(f"\n  TEST on {n_test} unseen mazes (greedy, no learning):")
    print(f"    random policy : success {base['success']:5.1f}% | avg reward {base['reward']:8.1f}")
    print(f"    trained agent : success {final['success']:5.1f}% | avg reward {final['reward']:8.1f} "
          f"| steps/optimal {final['ratio']:.2f}")
    np.save("q_table_random.npy", agent.Q)
    return agent, log, checkpoints, base, final


# ----------------------------------------------------------------------------
# 5. PLOTS
# ----------------------------------------------------------------------------
def plots(A, B, outdir="plots_random"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)

    if A is not None:
        curves_r, curves_s, rows = A
        for data, title, ylab, fname, scale in [
            (curves_r, "Exp A: reward per episode, mean over random mazes", "Reward", "A1_reward.png", 1),
            (curves_s, "Exp A: success rate, mean over random mazes", "% episodes reaching goal", "A2_success.png", 100),
        ]:
            sm = np.array([moving_avg(c * scale, 50) for c in data])
            mu, sd = sm.mean(0), sm.std(0)
            x = np.arange(49, 49 + len(mu))
            plt.figure(figsize=(8, 5))
            plt.plot(x, mu, color="tab:blue", label=f"mean of {len(data)} mazes (50-ep avg)")
            plt.fill_between(x, mu - sd, mu + sd, alpha=.25, color="tab:blue", label="± 1 std")
            plt.title(title); plt.xlabel("Episode"); plt.ylabel(ylab)
            plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(outdir, fname), dpi=150); plt.close()
        ratios = [r[3] for r in rows]
        plt.figure(figsize=(8, 5))
        plt.bar(range(1, len(ratios) + 1), [0 if np.isnan(v) else v for v in ratios], color="tab:green")
        plt.axhline(1, color="k", ls="--", label="optimal (1.0)")
        plt.title("Exp A: learnt path length / optimal (BFS) per maze")
        plt.xlabel("Maze #"); plt.ylabel("steps / optimal"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(outdir, "A3_path_optimality.png"), dpi=150); plt.close()

    if B is not None:
        agent, log, checkpoints, base, final = B
        w = max(50, len(log["success"]) // 100)
        plt.figure(figsize=(8, 5))
        plt.plot(np.arange(w - 1, len(log["success"])), moving_avg(log["success"], w) * 100,
                 label=f"training mazes ({w}-maze avg, includes exploration)", color="tab:blue")
        xs, ys = zip(*checkpoints)
        plt.plot(xs, ys, "o-", color="tab:red", label="UNSEEN mazes (greedy, no learning)")
        plt.axhline(base["success"], color="gray", ls=":", label=f"random policy ({base['success']:.0f}%)")
        plt.title("Exp B: one agent learning across many random mazes")
        plt.xlabel("Number of training mazes seen"); plt.ylabel("% mazes solved")
        plt.grid(alpha=.3); plt.legend(loc="center right"); plt.tight_layout()
        plt.savefig(os.path.join(outdir, "B1_generalisation_curve.png"), dpi=150); plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(np.arange(w - 1, len(log["reward"])), moving_avg(log["reward"], w), color="tab:purple")
        plt.title("Exp B: reward per training maze"); plt.xlabel("Training maze #"); plt.ylabel("Reward")
        plt.grid(alpha=.3); plt.tight_layout()
        plt.savefig(os.path.join(outdir, "B2_reward.png"), dpi=150); plt.close()

        plt.figure(figsize=(6, 5))
        vals = [base["success"], final["success"]]
        b = plt.bar(["Random policy", "Trained agent"], vals, color=["gray", "tab:green"])
        for rect, v in zip(b, vals):
            plt.text(rect.get_x() + rect.get_width() / 2, v + 1, f"{v:.0f}%", ha="center")
        plt.ylim(0, 110); plt.ylabel("% unseen mazes solved")
        plt.title("Exp B: test on 100 unseen mazes"); plt.tight_layout()
        plt.savefig(os.path.join(outdir, "B3_unseen_test.png"), dpi=150); plt.close()

        rng = np.random.default_rng(2024)
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
        for ax in axes:
            layout, opt = random_layout(rng)
            env = LocalObsMaze(layout)
            s, done, path = env.reset(), False, [env.pos]
            while not done:
                s, _, done, info = env.step(agent.act(s, greedy=True))
                path.append(env.pos)
            img = np.array([[{"#": 0, "T": 0.35, ".": 1, "S": 1, "G": 0.8}[ch] for ch in row] for row in layout])
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ys_, xs_ = zip(*path)
            ax.plot(xs_, ys_, "r-o", ms=3, lw=1.5)
            for r in range(N):
                for c in range(N):
                    if layout[r][c] == "T":
                        ax.text(c, r, "T", color="red", ha="center", va="center", fontweight="bold")
            ax.text(0, 0, "S", color="blue", ha="center", va="center", fontweight="bold")
            ax.text(N - 1, N - 1, "G", color="green", ha="center", va="center", fontweight="bold")
            ax.set_title(f"{'SOLVED' if info['success'] else 'FAILED'}: {env.steps} steps (optimal {opt})")
            ax.set_xticks([]); ax.set_yticks([])
        plt.suptitle("Exp B: trained agent on unseen mazes (black=wall, T=trap, red=path)")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "B4_sample_unseen_mazes.png"), dpi=150); plt.close()
    print(f"\nSaved plots in ./{outdir}/")


# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=["A", "B"], default=None)
    p.add_argument("--mazes", type=int, default=20, help="Exp A: number of random mazes")
    p.add_argument("--episodes", type=int, default=2000, help="Exp A: episodes per maze")
    p.add_argument("--train-mazes", type=int, default=40000, help="Exp B: training mazes")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-plots", action="store_true")
    a = p.parse_args()

    A = experiment_A(a.mazes, a.episodes, a.seed) if a.only in (None, "A") else None
    B = experiment_B(a.train_mazes, a.seed) if a.only in (None, "B") else None
    if not a.no_plots:
        plots(A, B)


if __name__ == "__main__":
    main()
