export type ExpState =
  | "proposed" | "approved" | "queued" | "running" | "analyzing" | "complete" | "failed" | "cancelled" | "rejected";
export type Verdict = "supported" | "refuted" | "inconclusive";
export type FindingStatus = "hypothesis" | Verdict | "replicated";

export interface HistoryEntry { state?: string; status?: string; at: string; actor: string; note?: string }
export interface Note { at: string; actor: string; text: string }

export interface RunBrief {
  id: string; seed: number; status: string; compute_hours: number; started_at: string | null;
  ended_at: string | null; bad: boolean; summary: RunSummary | null; progress?: string | null;
}

export interface Experiment {
  id: string; family: string; hypothesis: string; expected_if_true: string; expected_if_false: string;
  config: ExperimentConfig; state: ExpState; proposed_by: string; created_at: string; updated_at: string;
  verdict: Verdict | null; estimate_hours: number; compute_hours: number; replicates_finding_id: string | null;
  notes: Note[]; history: HistoryEntry[]; runs?: RunBrief[];
}

export interface ExperimentConfig {
  family: string; hypothesis: string; expected_if_true: string; expected_if_false: string;
  data: DataConfig; model: { n_layers: number; d_model: number; n_heads: number; context_len: number };
  training: { steps: number; batch_size: number; lr: number; checkpoint_every: number };
  probes: { targets: string[]; layers: "all" | number[]; checkpoints: "all" | "final" | number[];
    random_control: boolean; control_margin?: number; patching?: boolean; symbolic?: boolean };
  eval_sets?: { name: string; data: DataConfig }[];
  seeds: number[]; budget: { max_compute_hours: number };
}

export interface DataConfig {
  source: "rebound"; n_bodies: 2 | 3; n_systems: number; steps_per_orbit: number;
  mass_ratio_range: [number, number]; eccentricity_range?: [number, number]; data_seed?: number;
}

export interface TargetSummary { best_layer: number; r2: number; control_r2: number | null; control_spread?: number | null; required_margin?: number; beats_control: boolean; formed_at_step: number | null }
export interface RunSummary {
  final_val_mse?: number; baseline_cv_mse?: number; val_mse_ratio?: number; beats_baseline?: boolean;
  train_hours?: number; n_params?: number; device?: string; dataset_id?: string; stopped?: boolean;
  probes?: { final_checkpoint: number; control_margin: number; targets: Record<string, TargetSummary> };
  equations?: Record<string, { equation?: string; law_recovered?: boolean; fit_error_vs_true_law?: number; layer?: number; error?: string }>;
  patching?: { top_head: { layer: number; head: number; force_r2: number; effect: number }; control_head_effect: number };
  eval?: { name: string; val_mse_ratio: number; n_bodies: number; force_r2_best: number | null }[];
}

export interface Run {
  id: string; experiment_id: string; seed: number; status: string; started_at: string | null; ended_at: string | null;
  compute_hours: number; artifact_dir: string; git_commit: string | null; bad: boolean; notes: Note[];
  summary: RunSummary; device?: string; error?: string; progress?: string | null; dataset_id?: string;
}

export interface RunDetail extends Run {
  experiment: Experiment;
  equations: Record<string, Equations> | null; patching: Patching | null; eval: EvalResult[] | null;
  log: string; artifacts: { path: string; name: string; bytes: number }[]; artifacts_present: boolean;
}
export interface Equations {
  equation?: string; complexity?: number; loss?: number; fit_error_vs_probe?: number; fit_error_vs_true_law?: number;
  true_law_error_vs_probe?: number; law_recovered?: boolean; true_law?: string; layer?: number; probe_r2?: number; variables?: string[];
  pareto?: { complexity: number; loss: number; equation: string }[]; error?: string;
}
export interface Patching {
  head_scores: { layer: number; head: number; force_r2: number }[];
  patched: { layer: number; head: number; force_r2: number; clean_mse: number; patched_mse: number; effect: number; control: boolean }[];
  clean_mse: number;
}
export interface EvalResult {
  name: string; val_mse: number; baseline_cv_mse: number; val_mse_ratio: number; n_bodies: number;
  mass_ratio_range: number[]; probe_r2: Record<string, Record<string, number>>;
}

export interface Metric { step: number; name: string; value: number }
export interface ProbeRow {
  checkpoint_step: number; layer: number; target: string; r2: number;
  control_r2: number | null; // best of the random-init controls
  control_spread?: number | null; // max - min over the controls (null for single-control runs)
}
/** A probe counts if it beats the best random control by max(margin, control spread). Mirrors engine/probes/controls.py. */
export const requiredMargin = (margin: number, spread?: number | null) => Math.max(margin, spread ?? 0);
export const probeCounts = (r: Pick<ProbeRow, "r2" | "control_r2" | "control_spread">, margin: number) =>
  r.control_r2 !== null && r.r2 - r.control_r2 >= requiredMargin(margin, r.control_spread);

export interface Finding {
  id: string; claim: string; verdict: Verdict; status: FindingStatus; evidence_run_ids: string[];
  confidence_note: string; created_at: string; experiment_id: string | null; created_by: string;
  replicates: string | null; replications: { finding_id: string; agrees: boolean; run_ids: string[]; at: string }[];
  notes: Note[]; disputed: boolean; history: HistoryEntry[]; replication_count?: number;
}
export interface Question { id: string; text: string; status: "open" | "answered" | "dropped"; linked_finding_ids: string[]; created_at: string }
export interface PaperBrief { id: string; title: string; updated_at: string; cited_finding_ids: string[] }
export interface Paper extends PaperBrief {
  markdown: string; cite_overrides?: string[]; citation_warnings: string[];
  drafting: { status: string; section: string; error?: string } | null;
}

export interface Settings {
  daily_budget_hours: number; max_concurrent_runs: number; per_experiment_cap_hours: number;
  auto_approve_families: string[]; max_cycles_per_day: number; throughput: number | null;
}
export interface ReviewIssue {
  severity: "info" | "flag" | "pause"; category: string; message: string; source: "check" | "reviewer";
  finding_id?: string; run_id?: string;
}
export interface Review {
  id: string; created_at: string; since: string | null; verdict: "OK" | "FLAG" | "PAUSE"; summary: string;
  issues: ReviewIssue[]; status: "ok" | "open" | "addressed"; llm: boolean;
  responses: { issue: number; response: string; at: string; actor: string }[];
}

export interface LoopStatus {
  paused: boolean; paused_note?: string | null; settings: Settings; compute_used_today: number; cycles_today: number;
  last_cycle: { timestamp: string; type: string; payload: Record<string, unknown> } | null;
  worker: { online: boolean; last_seen_s: number | null };
}
export interface Dashboard {
  loop: LoopStatus; running_runs: (Run & { experiment: Experiment | null })[]; latest_findings: Finding[];
  open_questions: number; counts: Record<ExpState, number>; n_findings: number;
}
export interface DatasetMeta {
  id: string; config?: DataConfig; n_steps?: number; generated_seconds?: number; created_at?: string;
  used_by_runs?: string[]; status?: string;
}
export interface Hist { counts: number[]; edges: number[]; mean: number; std: number }
export interface DatasetStats {
  id: string; n_systems: number; n_steps: number; n_bodies: number;
  mass_ratio: Hist; period: Hist; eccentricity: Hist; semi_major_axis: Hist; energy: Hist;
}
export interface OrbitSample { index: number; masses: number[]; e: number; a: number; period: number; bodies: [number, number][][] }
export interface Template { name: string; config: ExperimentConfig }
export interface Validation { valid: boolean; errors: string[]; estimate_hours: number; auto_approvable: boolean; auto_reason: string }

export class ApiError extends Error {
  constructor(public status: number, public detail: unknown) {
    super(typeof detail === "string" ? detail : formatDetail(detail));
  }
}

function formatDetail(d: unknown): string {
  if (d && typeof d === "object" && "errors" in (d as Record<string, unknown>)) {
    return ((d as { errors: string[] }).errors ?? []).join("\n");
  }
  if (Array.isArray(d)) return d.map((x) => (x as { msg?: string }).msg ?? JSON.stringify(x)).join("\n");
  return JSON.stringify(d);
}

/** Hosted build (Vercel): GETs read the JSON snapshot that each agent cycle commits; nothing can be changed. */
export const STATIC = import.meta.env.VITE_STATIC === "1";
export const READ_ONLY_MSG = "This is a read-only snapshot. Run the lab locally (make dev) to make changes.";

export const exportUrl = (id: string, format: "markdown" | "latex") =>
  STATIC ? `/snapshot/api/papers/${id}/export.${format === "latex" ? "tex" : "md"}` : `/api/papers/${id}/export?format=${format}`;

export type SnapshotMeta = { generated_at: string; experiments: number; runs: number };
export const snapshotMeta = (): Promise<SnapshotMeta> => fetch("/snapshot/meta.json").then((r) => r.json());

async function req<T>(method: string, path: string, body?: unknown): Promise<T> {
  if (STATIC) {
    if (method !== "GET") throw new ApiError(405, READ_ONLY_MSG);
    const res = await fetch(`/snapshot/api${path.split("?")[0]}.json`);
    if (!res.ok) throw new ApiError(res.status, "not in the snapshot");
    return res.json() as Promise<T>;
  }
  const res = await fetch(`/api${path}`, {
    method,
    headers: body !== undefined ? { "Content-Type": "application/json" } : undefined,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    let detail: unknown = res.statusText;
    try { detail = (await res.json()).detail; } catch { /* not JSON */ }
    throw new ApiError(res.status, detail);
  }
  const ct = res.headers.get("content-type") ?? "";
  return (ct.includes("json") ? res.json() : res.text()) as Promise<T>;
}

export const api = {
  dashboard: () => req<Dashboard>("GET", "/dashboard"),
  loop: () => req<LoopStatus>("GET", "/loop"),
  startLoop: () => req<LoopStatus>("POST", "/loop/start", {}),
  pauseLoop: () => req<LoopStatus>("POST", "/loop/pause", {}),
  kill: () => req<LoopStatus & { cancelled: string[] }>("POST", "/loop/kill", {}),
  setBudget: (patch: Partial<Settings>) => req<LoopStatus>("PUT", "/loop/budget", patch),

  experiments: () => req<Experiment[]>("GET", "/experiments"),
  experiment: (id: string) => req<Experiment>("GET", `/experiments/${id}`),
  templates: () => req<Template[]>("GET", "/experiments/templates"),
  validate: (config: unknown) => req<Validation>("POST", "/experiments/validate", { config }),
  propose: (config: unknown, approve: boolean) => req<Experiment>("POST", "/experiments", { config, approve }),
  editConfig: (id: string, config: unknown) => req<Experiment>("PUT", `/experiments/${id}/config`, { config }),
  approve: (id: string, note = "") => req<Experiment>("POST", `/experiments/${id}/approve`, { note }),
  reject: (id: string, note = "") => req<Experiment>("POST", `/experiments/${id}/reject`, { note }),
  cancel: (id: string) => req<Experiment>("POST", `/experiments/${id}/cancel`, {}),

  runs: () => req<Run[]>("GET", "/runs"),
  run: (id: string) => req<RunDetail>("GET", `/runs/${id}`),
  metrics: (id: string) => req<Metric[]>("GET", `/runs/${id}/metrics`),
  probes: (id: string) => req<ProbeRow[]>("GET", `/runs/${id}/probes`),
  annotateRun: (id: string, body: { note?: string; bad?: boolean }) => req<Run>("POST", `/runs/${id}/annotate`, body),
  rerun: (id: string) => req<Experiment>("POST", `/runs/${id}/rerun`, {}),

  findings: () => req<Finding[]>("GET", "/findings"),
  promote: (id: string, status: string, note = "") => req<Finding>("POST", `/findings/${id}/promote`, { status, note }),
  annotateFinding: (id: string, body: { note?: string; disputed?: boolean }) => req<Finding>("POST", `/findings/${id}/annotate`, body),
  replicate: (id: string) => req<Experiment>("POST", `/findings/${id}/replicate`, {}),
  questions: () => req<Question[]>("GET", "/questions"),
  reviews: () => req<Review[]>("GET", "/reviews"),
  addQuestion: (text: string) => req<Question>("POST", "/questions", { text }),
  updateQuestion: (id: string, status: string) => req<Question>("PUT", `/questions/${id}`, { status }),

  papers: () => req<PaperBrief[]>("GET", "/papers"),
  paper: (id: string) => req<Paper>("GET", `/papers/${id}`),
  createPaper: (title: string) => req<Paper>("POST", "/papers", { title }),
  savePaper: (id: string, title: string, markdown: string, cite_overrides?: string[]) =>
    req<Paper>("PUT", `/papers/${id}`, { title, markdown, cite_overrides }),
  draftSection: (id: string, section: string, instructions: string, allow_unreplicated: string[]) =>
    req<{ status: string }>("POST", `/papers/${id}/draft-section`, { section, instructions, allow_unreplicated }),

  datasets: () => req<DatasetMeta[]>("GET", "/data/datasets"),
  datasetStats: (id: string) => req<DatasetStats>("GET", `/data/datasets/${id}/stats`),
  datasetSample: (id: string, n = 6, seed = 0) => req<OrbitSample[]>("GET", `/data/datasets/${id}/sample?n=${n}&seed=${seed}`),
  generateDataset: (data: DataConfig, n_steps: number) => req<{ id: string; status: string }>("POST", "/data/datasets", { data, n_steps }),
};
