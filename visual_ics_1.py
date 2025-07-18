# Supply‑Chain Team Play Environment  (rev‑B with shaping & smaller state)
# ================================================================
# Changes from rev‑A  (2025‑05‑31)
#   • Reward shaping via Ng‑Russell potential Φ(s)
#   • Deliver reward +100, process/QC +10  (stand out vs. −1 step cost)
#   • Hack probability lowered to 0.10 during training (raise for eval)
#   • trust_score capped at {0,1,2} to bound state space
#   • speed_mult removed from Q‑key (still simulated, but ignored for lookup)
#   • ε‑decay slowed to 0.999
#   • Extra diagnostics helpers (success‑rate, steps‑per‑delivery)
#
# Copy the whole cell into Colab; run the demo block at the bottom. (recommended)

import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


# ------------------------------------------------------------
# 1.  Environment with reward‑shaping
# ------------------------------------------------------------
class SupplyChainTeamEnv:
    """Multi‑agent 4‑stage supply chain with light cyber attacks + shaping."""
    def __init__(self, grid_size: int = 12, n_agents: int = 2,
                 hack_p: float = 0.10, max_steps: int = 250,
                 shaping_gamma: float = 0.97):
        self.grid_size = grid_size
        self.n_agents = n_agents
        self.hack_p = hack_p
        self.max_steps = max_steps
        self.shaping_gamma = shaping_gamma  # should match learner γ

        # Station coordinates
        self.raw = (0, 0)
        self.factory = (grid_size // 2, grid_size // 2)
        self.qc = (grid_size // 2, grid_size - 3)
        self.warehouse = (grid_size - 1, grid_size - 1)

        self.reset()

    # ---------------- util ----------------
    def _clip(self, v):
        return max(0, min(v, self.grid_size - 1))

    def _target_station(self, item_state):
        return [self.raw, self.factory, self.qc, self.warehouse][item_state]

    def _phi(self, i):
        """Potential Φ(s) = –Manhattan distance to next required station."""
        x, y = self.positions[i]
        tx, ty = self._target_station(self.item_states[i])
        return -(abs(x - tx) + abs(y - ty))

    def _state_tuple(self, i):
        x, y = self.positions[i]
        capped_trust = min(self.trust_scores[i], 2)  # 0,1,2+
        return (x, y, self.item_states[i], capped_trust)

    # ---------------- API ---------------
    def reset(self):
        self.positions, self.item_states = [], []
        self.speed_mults, self.done_flags, self.trust_scores = [], [], []
        for _ in range(self.n_agents):
            self.positions.append([random.randint(0, self.grid_size - 1),
                                   random.randint(0, self.grid_size - 1)])
            self.item_states.append(0)
            self.speed_mults.append(1)
            self.done_flags.append(False)
            self.trust_scores.append(0)
        self.t = 0
        return [self._state_tuple(i) for i in range(self.n_agents)]

    def step(self, actions):
        self.t += 1
        local_r, phi_old = [0.0]*self.n_agents, []
        for i in range(self.n_agents):
            phi_old.append(self._phi(i))

        # -------- movement + interaction --------
        for i in range(self.n_agents):
            if self.done_flags[i]:
                continue
            dx = dy = 0
            if actions[i] == 0:   dy = 1
            elif actions[i] == 1: dy = -1
            elif actions[i] == 2: dx = -1
            elif actions[i] == 3: dx = 1
            step = self.speed_mults[i]
            self.positions[i][0] = self._clip(self.positions[i][0] + dx*step)
            self.positions[i][1] = self._clip(self.positions[i][1] + dy*step)
            local_r[i] -= 1  # time cost

            x, y = self.positions[i]
            if self.item_states[i] == 0 and (x, y) == self.raw:
                self.item_states[i] = 1; local_r[i] += 5
            elif self.item_states[i] == 1 and (x, y) == self.factory:
                self.item_states[i] = 2; local_r[i] += 10
            elif self.item_states[i] == 2 and (x, y) == self.qc:
                self.item_states[i] = 3; local_r[i] += 10
            elif self.item_states[i] == 3 and (x, y) == self.warehouse:
                local_r[i] += 100; self.done_flags[i] = True

            # ----- cyber attack -----
            if (not self.done_flags[i]) and random.random() < self.hack_p:
                atk = random.choice(["speed_glitch","teleport","item_damage","obs_noise"])
                if atk == "speed_glitch":
                    self.speed_mults[i] = random.choice([0,2,3])
                elif atk == "teleport":
                    self.positions[i] = [random.randint(0, self.grid_size - 1),
                                          random.randint(0, self.grid_size - 1)]
                elif atk == "item_damage" and self.item_states[i] >= 1:
                    self.item_states[i] = max(1, self.item_states[i]-1)
                # obs_noise handled at agent side; ignored here
                detected = random.random() < 0.5
                if detected:
                    self.trust_scores[i] = min(self.trust_scores[i]+1, 2)
                    local_r[i] += 1
                else:
                    local_r[i] -= 2

        # -------- reward shaping --------
        for i in range(self.n_agents):
            phi_new = self._phi(i)
            shaping = self.shaping_gamma * phi_new - phi_old[i]
            local_r[i] += shaping

        global_r = sum(local_r)
        done = all(self.done_flags) or self.t >= self.max_steps
        return [self._state_tuple(i) for i in range(self.n_agents)], global_r, done


# ------------------------------------------------------------
# 2.  Q‑Learning (smaller Q‑key)
# ------------------------------------------------------------
class QLearningAgent:
    def __init__(self, acts, lr=0.1, gamma=0.97,
                 eps=1.0, eps_decay=0.999, eps_min=0.02):
        self.acts = acts; self.lr = lr; self.gamma = gamma
        self.eps = eps; self.eps_decay = eps_decay; self.eps_min = eps_min
        self.Q = defaultdict(lambda: np.zeros(len(acts)))

    def _k(self, s):
        # s = (x,y,item_state,trust)  → keep as‑is (no speed)
        return s

    def choose(self, s):
        k = self._k(s)
        if random.random() < self.eps:
            return random.choice(self.acts)
        return int(np.argmax(self.Q[k]))

    def learn(self, s, a, r, s2):
        k, k2 = self._k(s), self._k(s2)
        td = r + self.gamma * np.max(self.Q[k2]) - self.Q[k][a]
        self.Q[k][a] += self.lr * td

    def decay(self):
        self.eps = max(self.eps_min, self.eps * self.eps_decay)


# ------------------------------------------------------------
# 3.  Training loop & diagnostics
# ------------------------------------------------------------

def train(env_cfg={}, episodes=1000, n_agents=2):
    env = SupplyChainTeamEnv(n_agents=n_agents, **env_cfg)
    agents = [QLearningAgent([0,1,2,3,4]) for _ in range(n_agents)]
    returns, successes = [], []
    for ep in range(episodes):
        s = env.reset(); ep_r = 0; delivered = 0
        while True:
            a = [ag.choose(s[i]) for i, ag in enumerate(agents)]
            s2, G, done = env.step(a)
            for i, ag in enumerate(agents):
                ag.learn(s[i], a[i], G, s2[i])
            s = s2; ep_r += G
            if done:
                delivered = sum(env.done_flags)
                break
        for ag in agents: ag.decay()
        returns.append(ep_r)
        successes.append(delivered == n_agents)
        if ep % 100 == 0:
            print(f"Ep {ep}  Return {ep_r:6.1f}  succ {successes[-1]}  eps {agents[0].eps:.3f}")
    return env, agents, returns, successes


# ------------------------------------------------------------
# 4.  Plot helpers
# ------------------------------------------------------------

def plot_training(returns, successes):
    fig, ax = plt.subplots(1,2, figsize=(10,3))
    ax[0].plot(returns); ax[0].set_title("Team Return")
    ax[0].set_xlabel("Episode"); ax[0].grid(True)
    # success rate sliding window
    win = 50; sr = [np.mean(successes[max(0,i-win):i+1]) for i in range(len(successes))]
    ax[1].plot(sr); ax[1].set_title("Success Rate (win50)")
    ax[1].set_xlabel("Episode"); ax[1].set_ylim(0,1); ax[1].grid(True)
    plt.show()


def test_run(env, agents, steps=250):
    s = env.reset(); traj = [[] for _ in range(env.n_agents)]; rewards=[]
    for _ in range(steps):
        acts = [np.argmax(agents[i].Q[agents[i]._k(s[i])]) for i in range(env.n_agents)]
        s2, G, done = env.step(acts)
        rewards.append(G)
        for i in range(env.n_agents):
            traj[i].append(s[i][:2])
        s = s2
        if done: break
    return traj, rewards


def plot_traj(traj, env):
    g = env.grid_size; fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlim(-1,g); ax.set_ylim(-1,g); ax.set_xticks(range(g)); ax.set_yticks(range(g)); ax.grid()
    stations = {env.raw:"Raw", env.factory:"Fact", env.qc:"QC", env.warehouse:"Ship"}
    for (x,y),lab in stations.items():
        ax.scatter(x,y,marker='s',s=200); ax.text(x,y+0.3,lab,ha='center')
    colors=["tab:blue","tab:orange","tab:green","tab:red"]
    for i,tr in enumerate(traj):
        xs,ys=zip(*tr); ax.plot(xs,ys,marker='o',color=colors[i%len(colors)],label=f"A{i}")
    ax.legend(); ax.set_title("Trajectories"); plt.show()


# ------------------------------------------------------------
# 5.  Demo (training + test)
# ------------------------------------------------------------
if __name__ == "__main__":
    env, agents, R, S = train(env_cfg=dict(hack_p=0.10), episodes=800)
    plot_training(R,S)
    traj, rew = test_run(env, agents)
    print("Test episode rewards:", rew)
    plot_traj(traj, env)
