/**
 * Build CLI commands equivalent to the scheduler form submission.
 *
 * Used by the scheduler page to show a copyable CLI command above the submit button.
 */

function sq(s: string): string {
  if (!s) return '""';
  if (/^[a-zA-Z0-9._:/@=,+-]+$/.test(s)) return s;
  return `'${s.replace(/'/g, "'\\''")}'`;
}

const ALWAYS_ON_FLAGS = ["--side-info", "--hp-tuning", "--refined-initial-frame"];

// ---------------------------------------------------------------------------
// E2E (submit-session)
// ---------------------------------------------------------------------------

export interface SessionCliParams {
  backend: string;
  prompt: string;
  envId: string;
  totalTimesteps: number;
  seed: number;
  seeds: number[];
  maxIterations: number;
  passThreshold: number;
  numEnvs: number;
  model: string;
  vlmModel: string;
  numConfigs: number;
  coresPerRun: number;
  numEvals: number;
  useCodeJudge: boolean;
  device: string;
  thinkingEffort: string;
  criteriaDiagnosis?: boolean;
  motionTrailDual?: boolean;
  nodes: string[];
}

export function buildSessionCli(p: SessionCliParams): string {
  const a: string[] = ["uv run python -m p2p.scheduler job submit-session"];

  a.push(`--backend ${p.backend}`);
  a.push(`--prompt ${sq(p.prompt || "")}`);
  a.push(`--env-id ${p.envId}`);
  a.push(`--total-timesteps ${p.totalTimesteps}`);
  a.push(`--seed ${p.seed}`);
  if (p.seeds.length > 1) a.push(`--seeds ${p.seeds.join(",")}`);
  a.push(`--max-iterations ${p.maxIterations}`);
  a.push(`--pass-threshold ${p.passThreshold}`);
  a.push(`--num-envs ${p.numEnvs}`);
  if (p.model) a.push(`--model ${sq(p.model)}`);
  if (p.vlmModel) a.push(`--vlm-model ${sq(p.vlmModel)}`);
  if (p.numConfigs > 1) a.push(`--num-configs ${p.numConfigs}`);
  a.push(`--cores-per-run ${p.coresPerRun}`);
  if (p.numEvals !== 4) a.push(`--num-evals ${p.numEvals}`);

  a.push(...ALWAYS_ON_FLAGS);
  if (p.useCodeJudge) a.push("--use-code-judge");

  if (p.device !== "auto") a.push(`--device ${p.device}`);
  if (p.thinkingEffort) a.push(`--thinking-effort ${p.thinkingEffort}`);
  if (p.criteriaDiagnosis === false) a.push("--no-criteria-diagnosis");
  if (p.motionTrailDual === false) a.push("--no-motion-trail-dual");
  if (p.nodes.length > 0) a.push(`--nodes ${p.nodes.join(",")}`);

  return a.join(" \\\n  ");
}

// ---------------------------------------------------------------------------
// Benchmark (submit-benchmark)
// ---------------------------------------------------------------------------

export interface BenchmarkCliParams {
  backend: string;
  csvFile: string;
  totalTimesteps: number;
  seed: number;
  seeds: number[];
  maxIterations: number;
  passThreshold: number;
  numEnvs: number;
  numConfigs: number;
  model: string;
  vlmModel: string;
  maxParallel: number;
  coresPerRun: number;
  useCodeJudge: boolean;
  filterEnvs: string[];
  filterCategories: string[];
  filterDifficulties: string[];
  device: string;
  thinkingEffort: string;
  criteriaDiagnosis?: boolean;
  motionTrailDual?: boolean;
  nodes: string[];
}

export function buildBenchmarkCli(p: BenchmarkCliParams): string {
  const a: string[] = ["uv run python -m p2p.scheduler job submit-benchmark"];

  a.push(`--backend ${p.backend}`);
  if (p.csvFile) a.push(`--csv-file ${sq(`benchmark/${p.csvFile}`)}`);
  a.push(`--total-timesteps ${p.totalTimesteps}`);
  a.push(`--seed ${p.seed}`);
  if (p.seeds.length > 1) a.push(`--seeds ${p.seeds.join(",")}`);
  a.push(`--max-iterations ${p.maxIterations}`);
  a.push(`--pass-threshold ${p.passThreshold}`);
  a.push(`--num-envs ${p.numEnvs}`);
  if (p.model) a.push(`--model ${sq(p.model)}`);
  if (p.vlmModel) a.push(`--vlm-model ${sq(p.vlmModel)}`);
  if (p.numConfigs > 1) a.push(`--num-configs ${p.numConfigs}`);
  a.push(`--cores-per-run ${p.coresPerRun}`);
  if (p.maxParallel > 0) a.push(`--max-parallel ${p.maxParallel}`);

  a.push("--mode flat");

  // Filters
  if (p.filterEnvs.length > 0) a.push(`--filter-envs ${p.filterEnvs.join(",")}`);
  if (p.filterCategories.length > 0) a.push(`--filter-categories ${p.filterCategories.join(",")}`);
  if (p.filterDifficulties.length > 0) a.push(`--filter-difficulties ${p.filterDifficulties.join(",")}`);

  a.push(...ALWAYS_ON_FLAGS);
  if (p.useCodeJudge) a.push("--use-code-judge");

  if (p.device !== "auto") a.push(`--device ${p.device}`);
  if (p.thinkingEffort) a.push(`--thinking-effort ${p.thinkingEffort}`);
  if (p.criteriaDiagnosis === false) a.push("--no-criteria-diagnosis");
  if (p.motionTrailDual === false) a.push("--no-motion-trail-dual");
  if (p.nodes.length > 0) a.push(`--nodes ${p.nodes.join(",")}`);

  return a.join(" \\\n  ");
}
