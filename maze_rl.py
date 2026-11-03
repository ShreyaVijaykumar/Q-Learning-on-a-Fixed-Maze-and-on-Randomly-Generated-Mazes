"""
Maze Navigation with Reinforcement Learning (Q-Learning)
=========================================================
Agent      : robot that moves in a grid maze
Environment: 10x10 grid with walls, trap cells (pits) and a goal cell
State      : (row, col) position of the agent -> 100 discrete states
Actions    : 0=Up, 1=Right, 2=Down, 3=Left
Rewards    : +100 reaching goal | -50 falling in a trap | -5 hitting a wall/boundary
             -1 every normal step (encourages the shortest path)
Objective  : maximise expected cumulative (discounted) reward, i.e. reach the
             goal by the shortest safe path while avoiding traps.

Algorithm  : Tabular Q-Learning (off-policy TD control) with epsilon-greedy
             exploration and decaying epsilon.
             SARSA (on-policy) is trained too, purely as a comparison.

Libraries  : numpy, matplotlib only (no pre-built RL libraries, no dataset needed).

Usage:
    python maze_rl.py              # train, evaluate, save plots, animate path in terminal
    python maze_rl.py --episodes 3000 --seed 1
    python maze_rl.py --no-plots   # skip matplotlib windows/files
"""
import argparse
import time
import numpy as np

# ----------------------------------------------------------------------------
# 1. ENVIRONMENT
# ----------------------------------------------------------------------------
LAYOUT = [
    "S.........",
    ".###.####.",
    ".#...#..#.",
    ".#.###.##.",
    ".#.....#..",
    ".#####.#.#",
    ".....#.#..",
    ".###.#.##.",
    ".#T..#T#..",
    ".#..##...G",
]
# S=start, G=goal, #=wall, T=trap, .=free
ACTIONS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
ACTION_NAMES = ["Up", "Right", "Down", "Left"]
ARROWS = ["^", ">", "v", "<"]


class MazeEnv:
    R_GOAL, R_TRAP, R_WALL, R_STEP = 100.0, -50.0, -5.0, -1.0

    def __init__(self, layout=LAYOUT, max_steps=200):
        self.grid = [list(r) for r in layout]
        self.n_rows, self.n_cols = len(self.grid), len(self.grid[0])
        self.max_steps = max_steps
        self.n_states = self.n_rows * self.n_cols
        self.n_actions = 4
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                if self.grid[r][c] == "S":
                    self.start = (r, c)
                if self.grid[r][c] == "G":
                    self.goal = (r, c)
        self.reset()

    def _s(self, pos):  # (row, col) -> integer state id
        return pos[0] * self.n_cols + pos[1]

    def reset(self):
        self.pos = self.start
        self.steps = 0
        return self._s(self.pos)

    def step(self, action):
        dr, dc = ACTIONS[action]
        nr, nc = self.pos[0] + dr, self.pos[1] + dc
        self.steps += 1
        done, reward = False, self.R_STEP
        if not (0 <= nr < self.n_rows and 0 <= nc < self.n_cols) or self.grid[nr][nc] == "#":
            reward = self.R_WALL                      # bumped: stay in place
        else:
            self.pos = (nr, nc)
            cell = self.grid[nr][nc]
            if cell == "G":
                reward, done = self.R_GOAL, True
            elif cell == "T":
                reward, done = self.R_TRAP, True
        if self.steps >= self.max_steps:
            done = True                               # time-out
        return self._s(self.pos), reward, done, {"success": self.pos == self.goal}

    def render(self, path=()):
        out = []
        for r in range(self.n_rows):
            row = ""
            for c in range(self.n_cols):
                ch = self.grid[r][c]
                if (r, c) == self.pos:
                    ch = "A"
                elif (r, c) in path and ch == ".":
                    ch = "*"
                row += ch + " "
            out.append(row)
        return "\n".join(out)


# ----------------------------------------------------------------------------
# 2. AGENTS
# ----------------------------------------------------------------------------
class TabularAgent:
    """Q-table agent. algo='qlearning' (off-policy) or 'sarsa' (on-policy)."""

    def __init__(self, n_states, n_actions, algo="qlearning",
                 alpha=0.1, gamma=0.97, eps_start=1.0, eps_end=0.02,
                 eps_decay=0.995, rng=None):
        self.Q = np.zeros((n_states, n_actions))
        self.algo, self.alpha, self.gamma = algo, alpha, gamma
        self.eps, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay
        self.rng = rng or np.random.default_rng()
        self.n_actions = n_actions

    def act(self, s, greedy=False):
        if not greedy and self.rng.random() < self.eps:
            return int(self.rng.integers(self.n_actions))
        q = self.Q[s]
        return int(self.rng.choice(np.flatnonzero(q == q.max())))  # random tie-break

    def update(self, s, a, r, s2, done, a2=None):
        if done:
            target = r
        elif self.algo == "qlearning":
            target = r + self.gamma * self.Q[s2].max()
        else:                                          # SARSA
            target = r + self.gamma * self.Q[s2, a2]
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])   # TD update

    def decay(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)


# ----------------------------------------------------------------------------
# 3. TRAINING / EVALUATION
# ----------------------------------------------------------------------------
def train(env, agent, episodes=1500):
    log = {"reward": [], "steps": [], "success": [], "eps": []}
    for ep in range(episodes):
        s, total, done = env.reset(), 0.0, False
        a = agent.act(s)
        while not done:
            s2, r, done, info = env.step(a)
            a2 = agent.act(s2)
            agent.update(s, a, r, s2, done, a2)
            s, a, total = s2, a2, total + r
        agent.decay()
        log["reward"].append(total)
        log["steps"].append(env.steps)
        log["success"].append(1 if info["success"] else 0)
        log["eps"].append(agent.eps)
        if (ep + 1) % 250 == 0:
            w = slice(-100, None)
            print(f"  [{agent.algo:9s}] ep {ep+1:5d} | avg reward {np.mean(log['reward'][w]):8.2f}"
                  f" | success {np.mean(log['success'][w])*100:5.1f}% | eps {agent.eps:.3f}")
    return log


def evaluate(env, agent, episodes=100, random_policy=False, rng=None):
    """Greedy evaluation (no exploration, no learning)."""
    rng = rng or np.random.default_rng(0)
    rewards, steps, succ = [], [], []
    for _ in range(episodes):
        s, total, done = env.reset(), 0.0, False
        while not done:
            a = int(rng.integers(4)) if random_policy else agent.act(s, greedy=True)
            s, r, done, info = env.step(a)
            total += r
        rewards.append(total); steps.append(env.steps); succ.append(info["success"])
    return dict(avg_reward=np.mean(rewards), avg_steps=np.mean(steps),
                success_rate=100 * np.mean(succ))


def greedy_path(env, agent):
    s, done, path = env.reset(), False, [env.pos]
    while not done:
        s, _, done, info = env.step(agent.act(s, greedy=True))
        path.append(env.pos)
    return path, info["success"]


def moving_avg(x, w=50):
    x = np.asarray(x, float)
    return np.convolve(x, np.ones(w) / w, mode="valid")


# ----------------------------------------------------------------------------
# 4. PLOTS
# ----------------------------------------------------------------------------
def make_plots(env, ql, sarsa, ql_log, sa_log, path, show=False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    for log, name, col in [(ql_log, "Q-Learning", "tab:blue"), (sa_log, "SARSA", "tab:orange")]:
        ax[0, 0].plot(moving_avg(log["reward"]), label=name, color=col)
        ax[0, 1].plot(moving_avg(log["success"]) * 100, label=name, color=col)
        ax[1, 0].plot(moving_avg(log["steps"]), label=name, color=col)
    ax[0, 0].set(title="Cumulative reward per episode (50-ep moving avg)", xlabel="Episode", ylabel="Reward")
    ax[0, 1].set(title="Success rate (50-ep moving avg)", xlabel="Episode", ylabel="% episodes reaching goal")
    ax[1, 0].set(title="Steps per episode (50-ep moving avg)", xlabel="Episode", ylabel="Steps")
    ax[1, 0].set_yscale("log")
    for a in ax.flat[:3]:
        a.grid(alpha=.3); a.legend()

    # policy + value heatmap for Q-learning
    V = ql.Q.max(axis=1).reshape(env.n_rows, env.n_cols)
    mask = np.array([[ch == "#" for ch in row] for row in env.grid])
    ax[1, 1].imshow(np.ma.array(V, mask=mask), cmap="viridis")
    for r in range(env.n_rows):
        for c in range(env.n_cols):
            ch = env.grid[r][c]
            if ch == "#":
                ax[1, 1].add_patch(plt.Rectangle((c-.5, r-.5), 1, 1, color="black"))
            elif ch in "GT":
                ax[1, 1].text(c, r, ch, ha="center", va="center", color="red" if ch == "T" else "white",
                              fontsize=14, fontweight="bold")
            else:
                ax[1, 1].text(c, r, ARROWS[int(np.argmax(ql.Q[r*env.n_cols+c]))],
                              ha="center", va="center", color="white", fontsize=12)
    px, py = zip(*[(c, r) for r, c in path])
    ax[1, 1].plot(px, py, "r--", lw=1.5, alpha=.7)
    ax[1, 1].set(title="Learned policy (arrows), state values (colour), greedy path (red)")
    ax[1, 1].set_xticks([]); ax[1, 1].set_yticks([])
    plt.tight_layout()
    plt.savefig("learning_curves.png", dpi=130)
    print("Saved: learning_curves.png")
    if show:
        plt.show()



def save_individual_plots(ql_log, sa_log, outdir="plots"):
    """Save each learning curve as a separate PNG image in ./plots/"""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)

    specs = [
        ("reward",  "Cumulative Reward per Episode", "Reward", "1_reward_curve.png", False, 1),
        ("success", "Success Rate",                   "% episodes reaching goal", "2_success_rate.png", False, 100),
        ("steps",   "Steps per Episode",              "Steps", "3_steps_per_episode.png", True, 1),
    ]
    for key, title, ylabel, fname, logy, scale in specs:
        plt.figure(figsize=(8, 5))
        for log, name, col in [(ql_log, "Q-Learning", "tab:blue"), (sa_log, "SARSA", "tab:orange")]:
            raw = np.asarray(log[key], float) * scale
            plt.plot(raw, color=col, alpha=0.2)                                  # raw (faint)
            plt.plot(np.arange(49, len(raw)), moving_avg(raw), color=col, label=f"{name} (50-ep avg)")
        if logy:
            plt.yscale("log")
        plt.title(title); plt.xlabel("Episode"); plt.ylabel(ylabel)
        plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(outdir, fname), dpi=150); plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(ql_log["eps"], color="tab:green")
    plt.title("Exploration Rate (epsilon) Decay"); plt.xlabel("Episode"); plt.ylabel("epsilon")
    plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "4_epsilon_decay.png"), dpi=150); plt.close()
    print(f"Saved individual learning-curve images in ./{outdir}/")


# ----------------------------------------------------------------------------
# 5. MAIN / DEMO
# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=1500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--show", action="store_true", help="open plot window")
    p.add_argument("--animate", action="store_true", help="animate final path in terminal")
    args = p.parse_args()

    env = MazeEnv()
    print("=" * 60, "\nMAZE (S=start G=goal #=wall T=trap)\n" + "=" * 60)
    print(env.render())

    # Baseline: untrained random agent
    base = evaluate(env, None, random_policy=True, rng=np.random.default_rng(args.seed))
    print(f"\nBASELINE random policy  : success {base['success_rate']:.0f}% | "
          f"avg reward {base['avg_reward']:.1f} | avg steps {base['avg_steps']:.0f}")

    print(f"\nTRAINING for {args.episodes} episodes ...")
    ql = TabularAgent(env.n_states, 4, "qlearning", rng=np.random.default_rng(args.seed))
    sa = TabularAgent(env.n_states, 4, "sarsa", rng=np.random.default_rng(args.seed))
    ql_log = train(env, ql, args.episodes)
    sa_log = train(env, sa, args.episodes)

    print("\nEVALUATION (greedy policy, 100 episodes)")
    for name, ag in [("Q-Learning", ql), ("SARSA", sa)]:
        m = evaluate(env, ag)
        print(f"  {name:10s}: success {m['success_rate']:.0f}% | avg reward {m['avg_reward']:.1f} "
              f"| avg steps {m['avg_steps']:.1f}")

    path, ok = greedy_path(env, ql)
    print(f"\nQ-Learning greedy path: {len(path)-1} steps, reached goal = {ok}")
    env.reset()
    print(env.render(path=set(path)))

    if args.animate:
        env.reset()
        for pos in path:
            env.pos = pos
            print("\033[H\033[J" + env.render(set(path)))
            time.sleep(0.3)

    if not args.no_plots:
        make_plots(env, ql, sa, ql_log, sa_log, path, show=args.show)
        save_individual_plots(ql_log, sa_log)
    np.save("q_table.npy", ql.Q)
    print("Saved: q_table.npy")


if __name__ == "__main__":
    main()
