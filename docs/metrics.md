# Training Metrics Reference

All metrics are logged per rollout to `metrics/scalars.jsonl` and mirrored to TensorBoard (`tb_logs/`).

## Data Flow

```
Environment (64 envs)
  --> P2PCallback._on_step()      [per-step: episode returns, reward terms]
  --> P2PCallback._on_rollout_end() [per-rollout: aggregate + write]
        |
        +--> metrics/scalars.jsonl   (JSONL, consumed by dashboard)
        +--> tb_logs/                (TensorBoard events)
```

Metrics are written once per rollout (e.g., every 64 envs x 2048 steps = 131,072 timesteps).

## Viewing Metrics

- **Prompt2Policy Dashboard**: `http://localhost:3000` — auto-refreshes every 5s
- **TensorBoard**: `tensorboard --logdir runs/<session_id> --port 6006`

---

## Metric Groups

### Episodic Return

| Metric | Type | Description |
|--------|------|-------------|
| `episodic_return` | float | Mean total reward over the last 10 completed episodes. This is the primary training signal — if it goes up, the agent is learning. |

### Episode Stats

| Metric | Type | Description |
|--------|------|-------------|
| `episodic_return_std` | float | Standard deviation of returns over the last 10 episodes. High std means the agent performs inconsistently across episodes — it might succeed sometimes but fail others. A well-trained agent should have low std. |
| `episodic_return_min` | float | Worst return among the last 10 episodes. If min is much lower than mean, the agent has failure modes it hasn't fully resolved. |
| `episodic_return_max` | float | Best return among the last 10 episodes. If max is much higher than mean, the agent *can* perform well but doesn't do so reliably. |
| `episode_length` | float | Mean episode length over the last 10 episodes (in env steps). Sudden drops may indicate the agent is falling/dying early. Consistently hitting the max (1000 for MuJoCo) means the agent survives full episodes. |

### Reward Terms (Training)

| Metric | Type | Description |
|--------|------|-------------|
| `reward_term_{name}` | float | Per-component reward mean over the last 10 episodes. Keys are dynamic and depend on the reward function (e.g., `reward_term_forward_speed`, `reward_term_ctrl_penalty`). Use these to detect reward hacking — if one term dominates while others collapse, the agent is gaming the reward. |

### Loss

| Metric | Type | Description |
|--------|------|-------------|
| `policy_loss` | float | PPO clipped surrogate loss. Measures how much the policy changed this update. Should fluctuate but stay bounded — large spikes indicate unstable updates. |
| `value_loss` | float | Value function MSE loss. Measures how well the critic predicts future returns. Should decrease over training as the value function improves. |
| `entropy` | float | Entropy of the action distribution (exploration loss). Higher = more random exploration. PPO uses `ent_coef` to prevent premature convergence. Should decrease gradually — collapsing to zero too early means the agent stopped exploring. |

### PPO Diagnostic

| Metric | Type | Description |
|--------|------|-------------|
| `clip_fraction` | float | Fraction of samples where the PPO ratio was clipped by `clip_coef` (default 0.2). Values above 0.3 mean updates are too aggressive — the policy is changing too fast relative to the data. Consider lowering the learning rate. |
| `approx_kl` | float | Approximate KL divergence between old and new policy. PPO uses this as an early stopping signal. Values above 0.03-0.05 suggest the policy is diverging too quickly. |

### Policy & Gradient Health

| Metric | Type | Description |
|--------|------|-------------|
| `policy_std` | float | Mean standard deviation of the action distribution across all action dimensions. This is the raw exploration noise in action-space units. Should decrease gradually as the agent commits to a strategy. Collapse to near-zero = premature convergence. Stays high = agent isn't learning a decisive policy. |
| `grad_norm` | float | L2 norm of all policy gradients after the last training update. Spikes indicate gradient instability. Consistently large values may warrant lowering `max_grad_norm` or the learning rate. |

### General

| Metric | Type | Description |
|--------|------|-------------|
| `learning_rate` | float | Current learning rate (constant or scheduled). |
| `sps` | int | Steps per second — instantaneous throughput measured as a rolling average over the last 20 rollouts. Use this to monitor training speed and detect slowdowns. |

### Throughput

| Metric | Type | Description |
|--------|------|-------------|
| `rollout_time` | float | Seconds spent on data collection (environment stepping + policy inference). This is the "Worker" time. Dominated by CPU for MuJoCo environments. |
| `train_time` | float | Seconds spent on PPO gradient updates. This is the "Learner" time. For MLP policies, CPU is often faster than GPU (see `device` field). |
| `elapsed_time` | float | Cumulative wall-clock seconds since training started. |
| `device` | string | Compute device used for training (`cpu` or `cuda`). Logged alongside throughput so you can compare CPU vs GPU performance. |

### Eval (separate entries)

Eval entries are written at video capture checkpoints (configured by `num_evals`, default 4). They have `"type": "eval"` to distinguish them from training entries.

| Metric | Type | Description |
|--------|------|-------------|
| `total_reward` | float | Total reward for one deterministic eval episode. |
| `episode_length` | int | Eval episode length. |
| `reward_terms` | dict | Per-term reward breakdown (used by RewardChart and TermContributionBar on the dashboard). |

---

## TensorBoard Grouping

Metrics are organized into groups matching the dashboard layout:

| TensorBoard Group | Metrics |
|-------------------|---------|
| `return/` | episodic_return |
| `episode_stats/` | return_std, return_min, return_max, episode_length |
| `loss/` | clip_loss, value_loss, exploration_loss |
| `reward_terms/` | (dynamic, per reward function) |
| `ppo_diagnostic/` | clip_fraction, approx_kl |
| `health/` | policy_std, grad_norm |
| `general/` | learning_rate, sps |
| `throughput/` | rollout_time, train_time, elapsed_time |

## File Locations

```
runs/<session_id>/<iter_N>/
  metrics/
    scalars.jsonl        # All training + eval metrics (JSONL format)
  tb_logs/
    events.out.tfevents* # TensorBoard event files
  videos/
    eval_<step>.mp4      # Deterministic eval recordings
    frames/              # Extracted frames for VLM judgment
```
