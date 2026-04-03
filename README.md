<p align="center">
  <!-- TODO: logo image -->
  <h1 align="center">Prompt-to-Policy</h1>
</p>

<p align="center">
  <strong>Describe a behavior in a prompt. Get a trained policy.</strong><br/>
  LLM-powered reward engineering that writes, trains, judges, and iterates — until your RL agent does what you asked.
</p>

<p align="center">
  <a href="https://www.krafton.ai/blog/posts/2026-04-03-prompt-to-policy/prompt-to-policy_en.html"><img src="https://img.shields.io/badge/%F0%9F%8C%90%20Project-Page-4285F4?style=for-the-badge" alt="Project Page"/></a>
</p>

<div align="center">
  <img src="docs/demo_zoom_reveal.gif" alt="Prompt2Policy showcase: diverse learned behaviors from natural language intents" width="960"/>
</div>

## What It Does

| | Feature | Description |
|---|---|---|
| 🎯 | **Intent to Reward** | Describe behavior in natural language — LLM writes the reward function |
| 🏋️ | **Parallel Training** | PPO with multiple seeds and configs via Stable-Baselines3 |
| 👁️ | **Dual Judgment** | Code-based judge + VLM video judge evaluate trained policies |
| 🔄 | **Auto-Revision** | LLM diagnoses failures and rewrites reward + tunes hyperparameters |
| 🤖 | **Multi-LLM** | Claude, Gemini, GPT — any model with tool use support |
| 🦾 | **MuJoCo + IsaacLab** | 10 MuJoCo envs built-in, 90 IsaacLab envs optional |
| 📊 | **Dashboard** | Real-time web UI for sessions, training curves, rollout videos |

---

## Quick Start

### Install

```bash
git clone https://github.com/krafton-ai/Prompt2Policy.git
cd Prompt2Policy
uv sync --all-extras --python 3.11
```

> **Note:** Python 3.11 is required when using IsaacLab (Isaac Sim only ships cp311 wheels).
> MuJoCo-only users can use Python 3.12+, but 3.11 is recommended for compatibility.

<details>
<summary>Don't have uv?</summary>

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

See [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) for other platforms.

</details>

<details>
<summary>Headless Linux server? (AWS, cloud VMs)</summary>

Install system packages for rendering:

```bash
# Ubuntu 24.04+
sudo apt-get install -y xvfb libegl1 libgl1 libglu1-mesa

# Ubuntu 22.04
sudo apt-get install -y xvfb libegl1-mesa libgl1-mesa-glx libglu1-mesa
```

Set `MUJOCO_GL=egl` in your `.env` file. See the [User Guide](docs/GUIDE.md#headless-rendering-linux) for details.

</details>

<details>
<summary>Running parallel training on a GPU?</summary>

Enable [NVIDIA MPS](https://docs.nvidia.com/deploy/mps/index.html) so concurrent processes share the GPU efficiently instead of context-switching:

```bash
sudo nvidia-cuda-mps-control -d          # start MPS daemon
echo quit | sudo nvidia-cuda-mps-control  # stop when done
```

Recommended when running parallel training (`--max-parallel` > 1) or IsaacLab environments (GPU-vectorized). See the [User Guide](docs/GUIDE.md#nvidia-mps-optional) for details.

</details>

### Configure

```bash
cp .env.example .env
# Edit .env — set GEMINI_API_KEY (required), plus ANTHROPIC_API_KEY or OPENAI_API_KEY (optional)
```

### Run (Dashboard)

```bash
uv run uvicorn p2p.api.app:app --host 0.0.0.0 --port 8000 --reload --reload-dir src  # Terminal 1
cd frontend && npm install && npm run dev                                                    # Terminal 2
```

Open **http://localhost:3000**, enter an intent like *"do a backflip"*, and hit run. See the [dashboard tutorial](https://www.krafton.ai/blog/posts/2026-04-03-prompt-to-policy/prompt-to-policy_en.html/) for a video walkthrough. For CLI usage, see [CLI Reference](#cli-reference).

> **Remote server?** Create `frontend/.env.local` with `NEXT_PUBLIC_API_URL=http://<your-server-ip>:8000` so the browser can reach the API. See [Dashboard — Remote Access](docs/GUIDE.md#remote-access).

### Verify

```bash
uv run pytest tests/ -v
```

---

## Pipeline

<!-- TODO: pre-rendered SVG pipeline diagram -->

```
User Intent → Intent Elicitor → Reward Author + Judge Author
                                        ↓
                                   Code Review
                                        ↓
                              PPO Training (seeds × configs)
                                        ↓
                              Code Judge ∥ VLM Judge
                                        ↓
                                   Synthesizer
                                    ↓         ↓
                              [pass]  →  Done
                              [fail]  →  Revise Agent → next iteration
```

---

## Supported Environments

<details>
<summary><strong>MuJoCo (built-in)</strong> — 10 environments: all Gymnasium MuJoCo v5 locomotion</summary>

| Environment | DOF | Example Intents |
|-------------|-----|-----------------|
| **HalfCheetah-v5** | 6 | *"run forward fast"*, *"do a backflip"* |
| **Ant-v5** | 8 | *"walk in a circle"*, *"stand on rear legs"* |
| **Hopper-v5** | 3 | *"hop forward"*, *"jump as high as possible"* |
| **Walker2d-v5** | 6 | *"walk forward naturally"*, *"high knee sprinting"* |
| **Humanoid-v5** | 17 | *"walk with natural gait"*, *"perform a deep squat"* |
| **HumanoidStandup-v5** | 17 | *"stand up from the ground"* |
| **Swimmer-v5** | 2 | *"swim forward"*, *"swim in a zigzag"* |
| **Reacher-v5** | 2 | *"reach the target"* |
| **InvertedPendulum-v5** | 1 | *"keep the pole balanced"* |
| **InvertedDoublePendulum-v5** | 1 | *"balance both poles"* |

</details>

<details>
<summary><strong>IsaacLab (optional)</strong> — 90 environments: locomotion, manipulation, dexterous</summary>

[NVIDIA IsaacLab](https://github.com/isaac-sim/IsaacLab) environments are supported when Isaac Sim is installed.

| Category | Count | Examples |
|----------|-------|---------|
| Manipulation (Lift/Stack) | 21 | Franka lift/stack, Galbot, UR10 |
| Locomotion (Flat) | 12 | ANYmal B/C/D, Unitree Go1/Go2/A1, Cassie, Spot, H1, G1, Digit |
| Locomotion (Rough) | 11 | Same robots, rough terrain |
| Manipulation (Reach) | 8 | Franka, UR10, OpenArm |
| Humanoid | 8 | Humanoid locomotion variants |
| Assembly | 8 | AutoMate, Factory, Forge |
| Dexterous | 7 | Shadow hand, Allegro |
| Classic Control | 5 | Cartpole, Ant |
| Pick & Place | 4 | Franka, UR10 |
| Other | 6 | Quadcopter, Navigation |

**Requirements**: NVIDIA GPU with CUDA 12+, driver 525+, Ubuntu 22.04+.

</details>

---

## Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GEMINI_API_KEY` | **Yes** | — | Default LLM agent + VLM video judgment |
| `ANTHROPIC_API_KEY` | No | — | Required when using Claude models as LLM |
| `OPENAI_API_KEY` | No | — | Required when using GPT models as LLM |
| `MUJOCO_GL` | No | *(unset)* | Set to `egl` on headless Linux |

<details>
<summary>Advanced settings</summary>

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_HOST` | `localhost` | vLLM server host (local VLM inference) |
| `VLLM_PORT` | `8100` | vLLM server port |
| `VLLM_MODEL` | `Qwen/Qwen3.5-27B` | vLLM model name |

</details>

---

## CLI Reference

### E2E Loop

```bash
uv run python -m p2p.session.run_session \
  --session-id my_session \
  --prompt "do a backflip" \
  --loop-config '{"train": {"env_id": "HalfCheetah-v5", "total_timesteps": 1000000}, "max_iterations": 5, "pass_threshold": 0.7, "hp_tuning": true}'
```

### Benchmark

```bash
uv run python -m p2p.benchmark.benchmark_cli \
  --csv benchmark/test_cases_exotic_ant_halfcheetah_humanoid.csv \
  --max-iterations 5 \
  --total-timesteps 1000000 \
  --max-parallel 4 \
  --num-configs 3
```

See the [User Guide](docs/GUIDE.md) for full flag reference and API examples.

---

## Hardware

| | MuJoCo (default) | IsaacLab |
|---|-------------------|----------|
| **CPU** | 8+ cores (16+ recommended for parallel seeds) | 8+ cores |
| **RAM** | 16 GB (32+ recommended) | 32+ GB |
| **GPU** | Optional — CUDA GPU for EGL rendering | Required — 24+ GB VRAM (varies by task) |
| **Disk** | 20 GB | 100+ GB |

MuJoCo training is CPU-bound (PPO with MLP policy). A GPU accelerates headless rendering (EGL) and local VLM inference but is not required. IsaacLab environments are GPU-vectorized and need at least 24 GB VRAM.

---

## Development

```bash
uv run ruff check src/ tests/          # lint
uv run ruff format --check src/ tests/  # format
uv run pytest tests/ -v                 # test
cd frontend && npm run lint             # frontend lint
```

## Tech Stack

- **Training** — Gymnasium, MuJoCo, Stable-Baselines3, IsaacLab (optional)
- **LLM/VLM** — Anthropic Claude, Google Gemini, OpenAI GPT, vLLM
- **Backend** — FastAPI, uvicorn
- **Frontend** — Next.js, React, Tailwind CSS, Recharts, KaTeX
- **Dev** — uv, ruff, pytest

## Documentation

- [User Guide](docs/GUIDE.md) — detailed setup, usage, intent tips, LLM models, IsaacLab installation
- [Architecture](docs/ARCHITECTURE.md) — code-level module map and execution flow
- [v1.0 Release Notes](docs/v1-release-notes.html) — known limitations and roadmap

---

## Citation

```bibtex
@misc{prompt2policy2026,
  title   = {Prompt-to-Policy: Agentic Engineering for Reinforcement Learning},
  author  = {{KRAFTON AI} and {Ludo Robotics} and Wooseong Chung and Taegwan Ha and Yunhyeok Kwak and Taehwan Kwon and Jeong-Gwan Lee and Kangwook Lee and Suyoung Lee},
  year    = {2026},
  url     = {https://github.com/krafton-ai/Prompt2Policy}
}
```

## License

This project is licensed under the [MIT License](LICENSE).

<p align="center">
  <em>Whether you're an RL researcher tired of hand-tuning rewards or a newcomer who just wants to describe a behavior and get a trained policy — this is for you.</em>
</p>
