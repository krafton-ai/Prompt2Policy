# Related Work

Prompt2Policy belongs to a growing family of systems that use LLMs to automate parts of the ML research cycle. We position it below relative to reward design systems, automated research frameworks, and algorithm discovery.

## LLM-Driven Reward Design

| | Evaluation | Iteration Memory | Revision Scope | Autonomous? |
|---|---|---|---|---|
| [**Eureka**](https://arxiv.org/abs/2310.12931) (Ma et al., ICLR 2024) | Pre-defined fitness function F | Last-best reward reflection only | Reward code | Partially — F must be hand-defined; optional Eureka-HF mode needs human text feedback |
| [**Text2Reward**](https://arxiv.org/abs/2309.11489) (Xie et al., ICLR 2024) | Environment success metrics | Last code only (no accumulated history) | Reward code | Partially — zero-shot generation is autonomous; optional refinement loop requires human feedback |
| [**DrEureka**](https://arxiv.org/abs/2406.01967) (Ma et al., RSS 2024) | Eureka fitness F | Last-best reflection (reward stage); none (DR stage) | Reward + domain randomization | Yes, but DR stage is single-pass (no closed-loop DR refinement) |
| [**CARD**](https://arxiv.org/abs/2410.14660) (Sun et al., KBS 2025) | Trajectory Preference Evaluation (TPE) | Feedback history in context | Reward code | Yes — TPE can skip redundant training in later iterations by pre-screening against prior trajectories |
| [**ReEvo**](https://arxiv.org/abs/2402.01145) (Ye et al., NeurIPS 2024) | Task-specific fitness | Short-term + long-term verbal reflections | Heuristic code | Yes — but targets combinatorial optimization, not RL |
| [**RLZero**](https://arxiv.org/abs/2412.05718) (Sikchi et al., NeurIPS 2025) | GPT-4o win rates + InternVideo2 similarity | None (zero-shot) | N/A (reward-free) | Yes — bypasses reward design entirely via video imagination + BFM |
| **Prompt2Policy** | **Self-generated Code Judge + VLM video analysis** | **Lineage tree with tiered lessons (HARD/STRONG/SOFT)** | **Reward code + hyperparameters + rebase to any past iteration** | **Yes — fully autonomous, any natural-language intent** |

## Automated ML Research & Algorithm Discovery

| | Search Target | Search Structure | Memory Mechanism |
|---|---|---|---|
| [**AutoResearch**](https://github.com/karpathy/autoresearch) (Karpathy, 2026) | Full training code (arch + optimizer + HP) | Linear ratchet (keep/revert) | `results.tsv` + git history |
| [**AI Scientist v2**](https://arxiv.org/abs/2504.08066) (Yamada et al., 2025) | ML research ideas + code | Best-first tree search with backtracking | Tree node checkpoints |
| [**POISE**](https://arxiv.org/abs/2603.23951) (Xia et al., arXiv 2026) | LLM policy optimization algorithms | Epistemic evolutionary search | Genealogical archive + mechanism-level reflections |
| [**EvoX**](https://arxiv.org/abs/2301.12457) (Huang et al., IEEE TEVC 2024) | General optimization (including policy parameters) | Population-based evolutionary | Hierarchical state management (algorithm-specific, e.g., CMA-ES covariance) |
| **Prompt2Policy** | **Reward functions for RL policies** | **Agentic tool-calling loop** | **Experiment lineage tree + accumulated lessons** |

## How Prompt2Policy Differs

**Open-ended intent evaluation.**
Eureka and ReEvo require a pre-defined fitness function F that humans must provide per task; Text2Reward replaces F with human feedback on rollout videos, which avoids the need for a formal metric but introduces a per-iteration human dependency. Prompt2Policy eliminates both requirements: given only a natural-language intent (e.g., *"do a backflip"*), the system generates both the reward function and its own evaluation criteria — a Code Judge that scores trajectories plus a VLM that watches rollout video. This makes Prompt2Policy applicable to any describable behavior without per-task engineering or human-in-the-loop evaluation.

**Agentic information gathering.**
Most systems provide fixed context to the LLM each iteration (Eureka includes per-component reward reflection data; AutoResearch's agent can read `results.tsv` and git history). Prompt2Policy's Judge and Revise agents use tool-calling to actively query what they need: past iterations' reward code, training dynamics, checkpoint-level rollout judgments, cross-config comparisons, and the experiment lineage. The agent decides what information is relevant, rather than receiving a one-size-fits-all prompt. This scales to longer campaigns where injecting all history into the prompt would exceed context limits.

**Experiment lineage with tiered lessons.**
POISE's genealogically linked archive with epistemic acquisition functions addresses a related challenge in algorithm discovery — tracking candidate lineage with mechanism-level reflections to guide search. Prompt2Policy takes a complementary approach in the reward design domain, adding graduated enforcement tiers: HARD lessons (catastrophic failures) are never violated, STRONG lessons guide default behavior, SOFT lessons can be freely challenged, and RETIRED lessons remain visible to prevent rediscovery. During detected plateaus, the Revise agent can demote or retire lessons to escape local optima. By contrast, AutoResearch deliberately uses a minimal memory model (flat `results.tsv` + git history), and Eureka carries forward only the single best reward and its reflection — simpler designs that trade structured institutional memory for operational simplicity.

**Joint reward-hyperparameter search with rebasing.**
Eureka and Text2Reward search only the reward function space. Prompt2Policy jointly revises reward code and training hyperparameters (learning rate, entropy coefficient, rollout length, etc.) in a single coordinated proposal, informed by training dynamics analysis. The tree structure allows the Revise agent to rebase onto any past iteration — not just the previous or best one — enabling non-linear exploration similar to AI Scientist v2's tree search but within the reward design domain.
