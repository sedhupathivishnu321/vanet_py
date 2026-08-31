"""A small DQN signal controller (spec section 30 -- "one computationally
practical RL baseline").

State  s = [q_main/10, q_cross/10, phase, phase_elapsed/max_green, mean_speed/15]
Action a in {0, 1} = desired phase.
Reward r = -(q_main + q_cross) - 0.1 * switch_penalty  per decision step.

Training happens against the IDM micro-simulator in ``train_dqn`` and the
learned greedy policy is wrapped by ``DQNController``.
"""
from __future__ import annotations

import copy
import random
from collections import deque

import numpy as np


def _torch():
    import torch
    return torch


class _QNet:
    def __init__(self, in_dim=5, hidden=64, out_dim=2, lr=5e-4, seed=0):
        torch = _torch()
        import torch.nn as nn
        torch.manual_seed(seed)
        self.torch = torch
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

    def q(self, s):
        return self.net(self.torch.as_tensor(np.asarray(s, np.float32)))

    def act(self, s, eps=0.0):
        if random.random() < eps:
            return random.randint(0, 1)
        with self.torch.no_grad():
            return int(self.q(s).argmax().item())


class DQNController:
    def __init__(self, cfg, weights_path: str | None = None, qnet: "_QNet | None" = None):
        self.cfg = cfg
        self.max_green = float(cfg["sumo"]["traffic_light"]["max_green_s"])
        self.qnet = qnet
        if qnet is None and weights_path:
            self.load(weights_path)

    def _state(self, ctx):
        hist = ctx.get("state_history")
        spd = float(np.mean(hist[-1, :, 0])) if hist is not None and len(hist) else 12.0
        return [ctx.get("queue_main", 0) / 10.0, ctx.get("queue_cross", 0) / 10.0,
                float(ctx.get("phase", 0)),
                min(ctx.get("phase_elapsed", 0) / self.max_green, 2.0),
                spd / 15.0]

    def __call__(self, ctx) -> int:
        if self.qnet is None:
            # untrained -> behave like max pressure
            return 0 if ctx.get("queue_main", 0) >= ctx.get("queue_cross", 0) else 1
        return self.qnet.act(self._state(ctx), eps=0.0)

    def save(self, path):
        torch = _torch()
        torch.save(self.qnet.net.state_dict(), path)

    def load(self, path):
        torch = _torch()
        self.qnet = _QNet()
        self.qnet.net.load_state_dict(torch.load(path, map_location="cpu"))
        self.qnet.net.eval()

    def reset(self):
        pass


def train_dqn(cfg, corridor: dict, demand_level: str, seed: int = 42,
              logger=None) -> DQNController:
    """Tabular-free DQN trained by repeatedly running the IDM simulator with an
    epsilon-greedy phase policy."""
    torch = _torch()
    from src.sumo.idm_sim import IDMSimulator
    dcfg = cfg["control"]["dqn"]
    episodes = int(dcfg["episodes"])
    gamma = float(dcfg["gamma"])
    eps = 1.0
    q = _QNet(lr=float(dcfg["lr"]), seed=seed)
    target = copy.deepcopy(q.net)
    buf = deque(maxlen=int(dcfg["replay_size"]))
    rng = random.Random(seed)

    for ep in range(episodes):
        transitions = []

        class _Policy:
            def __init__(self): self.prev = None
            def __call__(self, ctx):
                s = DQNController(cfg, qnet=q)._state(ctx)
                a = q.act(s, eps=eps)
                r = -(ctx.get("queue_main", 0) + ctx.get("queue_cross", 0))
                if self.prev is not None:
                    ps, pa, pr = self.prev
                    r_pen = pr - (0.5 if pa != a else 0.0)
                    transitions.append((ps, pa, r_pen, s, False))
                self.prev = (s, a, r)
                return a

        sim = IDMSimulator(cfg, corridor, demand_level, seed + ep, controller=_Policy())
        out = sim.run()
        for tr in transitions:
            buf.append(tr)

        # learn
        if len(buf) >= 64:
            for _ in range(20):
                batch = rng.sample(buf, 64)
                s, a, r, s2, d = zip(*batch)
                s = torch.as_tensor(np.array(s, np.float32))
                s2 = torch.as_tensor(np.array(s2, np.float32))
                a = torch.as_tensor(np.array(a)).long()
                r = torch.as_tensor(np.array(r, np.float32))
                qv = q.net(s).gather(1, a[:, None]).squeeze(1)
                with torch.no_grad():
                    tv = r + gamma * target(s2).max(1).values
                loss = torch.nn.functional.smooth_l1_loss(qv, tv)
                q.opt.zero_grad(); loss.backward(); q.opt.step()
        if ep % 3 == 0:
            target.load_state_dict(q.net.state_dict())
        eps = max(0.05, eps * float(dcfg["epsilon_decay"]))
        if logger:
            logger.info(f"  DQN ep {ep:3d}  reward~{np.mean([t[2] for t in transitions]):.2f}  eps={eps:.2f}")
    q.net.eval()
    return DQNController(cfg, qnet=q)
