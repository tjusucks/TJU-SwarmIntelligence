### LLM-Swarm: Multi-Robot Cooperative Transport

This repository currently provides a trainable MARL baseline for cooperative object transport.
The training pipeline uses:

- PettingZoo parallel environment wrapper.
- Shared-policy IPPO trainer (homogeneous robots).
- Existing physics simulator in src/sim.


What is implemented now

- Training script: scripts/train_ippo.py
- Environment wrapper: src/envs/transport_parallel_env.py
- IPPO trainer: src/agents/ippo.py
- Interactive simulator: src/main.py


1. Environment setup

Option A: Conda environment (recommended for local experiments)

1) Create and activate env

```bash
conda create -n robot_sim python=3.10 -y
conda activate robot_sim
```

2) Install project dependencies from pyproject.toml

```bash
pip install -e .
```

Option B: uv workflow

1) Install dependencies

```bash
uv sync
```

2) Run commands with uv

```bash
uv run python scripts/train_ippo.py --help
```

2. Sanity check before training

From project root:

```bash
cd llm_swarm
```

Run simulator:

```bash
python -m src.main
```

If window opens and robots move, simulation core works.


3. Quick smoke training

Use a short run to verify end-to-end training and checkpoint saving:

```bash
python scripts/train_ippo.py --total-steps 512 --rollout-steps 128 --max-episode-steps 400 --force-max 500 --stuck-patience 300 --log-interval 1 --save-path checkpoints/smoke_ippo.pt
```

Expected output includes lines like:

update=1 steps=128 avg_return_100=... actor_loss=... critic_loss=... entropy=...

At the end, a checkpoint is saved to checkpoints/smoke_ippo.pt.


4. Full training commands

Fixed map baseline:

```bash
python scripts/train_ippo.py --level fixed --seed 42 --total-steps 200000 --rollout-steps 1024 --max-episode-steps 1200 --learning-rate 3e-4 --update-epochs 10 --minibatch-size 256 --device cpu --save-path checkpoints/ippo_fixed.pt
```

Mild randomization (better generalization):

```bash
python scripts/train_ippo.py --level mild --seed 42 --total-steps 400000 --rollout-steps 1024 --max-episode-steps 1200 --learning-rate 3e-4 --update-epochs 10 --minibatch-size 256 --device cpu --save-path checkpoints/ippo_mild.pt
```

Use GPU if available:

```bash
python scripts/train_ippo.py --level mild --device cuda --save-path checkpoints/ippo_cuda.pt
```

Full random maps (recommended default training setup):

```bash
python scripts/train_ippo.py --level full --seed 42 --total-steps 400000 --rollout-steps 1024 --max-episode-steps 2400 --force-max 500 --stuck-patience 600 --device cpu --save-path checkpoints/ippo_full.pt
```

5. Key arguments

- --level: fixed, mild, moderate, full
- --total-steps: total environment steps
- --rollout-steps: on-policy rollout length before each PPO update
- --max-episode-steps: per-episode truncation limit
- --force-max: max force magnitude mapped from policy action
- --stuck-patience: steps allowed in near-stuck condition before failure
- --stuck-move-eps: movement threshold used for stuck detection
- --learning-rate: Adam learning rate
- --update-epochs: PPO optimization epochs per rollout
- --minibatch-size: PPO minibatch size
- --device: cpu or cuda
- --save-path: checkpoint output path
- --log-interval: print interval in update units


6. Checkpoint content

Saved checkpoint includes:

- model: policy/value network weights
- config: IPPO hyper-parameters
- obs_dim, action_dim
- agent_order


7. Current observation/action design

- Action per robot: 2D continuous force command in range [-1, 1], mapped to cmd_fx and cmd_fy.
- Observation: global map style vector (object state, goal relation, all robots, padded obstacle list) + own agent index.
- Reward: shared dense progress reward + small step penalty + blocked-ratio penalty + success/failure terminal bonus.
- Failure criterion: episode fails only when the system is stuck for a patience window (all blocked or invalid static state), or times out at max episode steps.


8. Troubleshooting

Issue: ModuleNotFoundError: No module named src

- Use either:
	- python -m src.main
	- python scripts/train_ippo.py
- Or run from project root first:
	- cd llm_swarm

Issue: Torch or PettingZoo import error

- Reinstall dependencies:
	- pip install -e .

Issue: Slow training on CPU

- Reduce rollout and total steps for debugging.
- Use --device cuda if GPU is available.


9. Minimal reproducible experiment recipe

1) Smoke test:

```bash
python scripts/train_ippo.py --total-steps 512 --rollout-steps 128 --max-episode-steps 200 --log-interval 1 --save-path checkpoints/smoke_ippo.pt
```

2) Baseline run:

```bash
python scripts/train_ippo.py --level fixed --seed 42 --total-steps 200000 --save-path checkpoints/ippo_fixed.pt
```

3) Generalization run:

```bash
python scripts/train_ippo.py --level mild --seed 42 --total-steps 400000 --save-path checkpoints/ippo_mild.pt
```

10. Evaluate a checkpoint

Run deterministic policy evaluation with summary metrics:

```bash
python scripts/eval_ippo.py --ckpt checkpoints/ippo_fixed.pt --episodes 50 --level full --max-episode-steps 2400 --force-max 500 --stuck-patience 600 --device cpu
```

If your checkpoint is outside the project directory, use an absolute path:

```bash
python scripts/eval_ippo.py --ckpt /path/to/ippo_fixed.pt --episodes 50 --level full --device cpu
```

GPU evaluation:

```bash
python scripts/eval_ippo.py --ckpt checkpoints/ippo_fixed.pt --episodes 50 --level full --device cuda
```

Visual replay switch in the same eval script:

```bash
python scripts/eval_ippo.py --ckpt checkpoints/ippo_fixed.pt --episodes 3 --level full --device cpu --render --fps 60
```

11. Replay controls

Keyboard controls during replay:

- SPACE: pause or resume.
- R: reset current episode.
- Window close button: exit replay.
