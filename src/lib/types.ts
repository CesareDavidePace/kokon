// Mirrors the qls-service JSON contract (crates/qls-core).

export type Mode = "auto" | "generate" | "edit" | "inpaint" | "outpaint" | "multi_reference" | "upscale";
export type ResolvedMode = Exclude<Mode, "auto">;
export type Quality = "fast" | "balanced" | "quality";
export type ResolutionMode = "auto" | "source" | "1k" | "1.5k" | "2k" | "custom";
export type OutputFormat = "png" | "jpeg" | "webp";
export type Units = "pixels" | "percent";
/** Effort of a Detailed upscale ("off" is legacy and behaves like fast). */
export type RefineMode = "off" | "fast" | "best";
export type UpscaleMethod = "quick" | "detailed";

export interface UpscaleSpec { method: UpscaleMethod; scale?: number | null; width?: number | null; height?: number | null }

export interface Size { width: number; height: number }
export interface Rect { x: number; y: number; width: number; height: number }

export interface OutpaintSpec { left: number; right: number; top: number; bottom: number; units: Units }

export interface Advanced {
  steps?: number | null;
  cfg?: number | null;
  sampler?: string | null;
  scheduler?: string | null;
  strength?: number | null;
  batch_count?: number | null;
  output_format?: OutputFormat | null;
  vae_tiling?: boolean | null;
  refine?: RefineMode | null;
  /** Keep a HEIC source's HDR gain map (Ultra HDR JPEG); null = on. */
  hdr?: boolean | null;
}

export interface GenerationRequest {
  prompt: string;
  negative_prompt?: string | null;
  mode: Mode;
  images: string[];
  mask?: string | null;
  outpaint?: OutpaintSpec | null;
  quality: Quality;
  resolution: ResolutionMode;
  aspect_ratio?: string | null;
  width?: number | null;
  height?: number | null;
  seed?: number | null;
  preserve_dimensions?: boolean;
  advanced: Advanced;
  client?: string;
  parent_id?: string | null;
  upscale?: UpscaleSpec | null;
}

export interface WorkingCanvas { source: Size; content: Size; canvas: Size; content_offset: [number, number]; scale: number }
export interface RefPlan { index: number; source: Size; canvas: WorkingCanvas }

export interface Plan {
  mode: ResolvedMode;
  auto_mode: boolean;
  mode_reason: string;
  output: Size;
  working: Size;
  content: Rect;
  steps: number;
  cfg: number;
  sampler: string;
  scheduler: string | null;
  strength: number | null;
  batch_count: number;
  vae_tiling: boolean;
  references: RefPlan[];
  aspect_label: string;
  notes: string[];
  optimized_for_vram: boolean;
  estimated_seconds: number;
  cache_threshold: number | null;
  upscaler: string | null;
  upscale: { method: UpscaleMethod; source: Size } | null;
}

export interface AppErrorBody { code: string; message: string; suggestion?: string; detail?: string }

export interface ReferenceRecord { index: number; input_id: string; original_path?: string | null; width: number; height: number }

export interface PipelineRecord {
  base: Size | null;
  refine: RefineMode | null;
  refine_steps: number | null;
  refine_denoise: number | null;
  hires_upscaler: string | null;
  cache_threshold: number | null;
  enlarged_with: string | null;
  enlarged_from: Size | null;
}

export interface GenerationRecord {
  id: string;
  created_at: string;
  mode: ResolvedMode;
  prompt: string;
  negative_prompt: string;
  seed: number;
  width: number;
  height: number;
  working_width: number;
  working_height: number;
  model: string;
  sampler: string;
  scheduler: string | null;
  steps: number;
  cfg: number;
  strength: number | null;
  quality: Quality;
  resolution: ResolutionMode;
  references: ReferenceRecord[];
  mask_input_id: string | null;
  outpaint: { left: number; right: number; top: number; bottom: number } | null;
  file_path: string;
  extra_paths: string[];
  favorite: boolean;
  parent_id: string | null;
  duration_ms: number;
  notes: string[];
  request: GenerationRequest | null;
  pipeline?: PipelineRecord | null;
}

export type JobStatus = "queued" | "starting_backend" | "preparing" | "running" | "saving" | "done" | "failed" | "cancelled";
export type BackendStage = "preparing" | "encoding_images" | "encoding_prompt" | "generating" | "upscaling" | "refining" | "decoding" | "finishing";

export interface JobView {
  id: string;
  status: JobStatus;
  stage: BackendStage | null;
  step: [number, number] | null;
  image: [number, number] | null;
  seconds_per_step: number | null;
  queue_position: number;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  prompt: string;
  plan: Plan | null;
  seed: number | null;
  results: GenerationRecord[];
  error: AppErrorBody | null;
  notes: string[];
  detailed_progress: boolean;
}

export interface InputRecord {
  id: string;
  path: string;
  original_name: string | null;
  original_path: string | null;
  width: number;
  height: number;
  format: string;
  has_alpha: boolean;
  /** HEIC with an HDR gain map. */
  hdr?: boolean;
}

export type ServerState = "stopped" | "starting" | "ready" | "error";

export interface StatusResponse {
  service: { version: string; pid: number; uptime_s: number };
  server: { state: ServerState; owned: boolean; pid: number | null; port: number; model: string | null; model_loaded: boolean; error: AppErrorBody | null };
  profile: { id: string; display_name: string };
  gpu: { name: string; memory_total_mib: number; memory_used_mib: number; memory_free_mib: number; driver_version: string } | null;
  active_job: JobView | null;
  queue: { queued: number; jobs: JobView[] };
  upscaler?: { model: string; available: boolean; refine_mode: RefineMode };
  setup_complete: boolean;
}

export interface Check { id: string; label: string; ok: boolean; detail: string }
export interface SetupReport { ready: boolean; checks: Check[]; devices: [string, string][]; backend_version: string | null }

export type TextEncoderPlacement = "ram" | "gpu" | "cpu" | "auto";

export interface ModelProfile {
  id: string;
  displayName: string;
  diffusionModel: string;
  vae: string;
  textEncoder: string;
  visionProjector: string | null;
  backendArgs: string[];
  recommendedSettings: Record<string, unknown>;
  capabilities: Record<string, boolean>;
}

export interface AppConfig {
  schemaVersion: number;
  sdcppDir: string;
  serverPort: number;
  servicePort: number;
  activeProfile: string;
  profiles: ModelProfile[];
  device: string;
  textEncoderPlacement: TextEncoderPlacement;
  vramReserveGib: number;
  flashAttention: boolean;
  sageAttention: boolean;
  vaeTileSize: number;
  prefixCacheType: string;
  eagerLoad: boolean;
  upscalersDir: string | null;
  upscalerModel: string;
  upscaleTileSize: number;
  refineMode: RefineMode;
  extraServerArgs: string[];
  outputDir: string;
  keepServerRunning: boolean;
  idleUnloadMinutes: number;
  setupComplete: boolean;
  ui: Record<string, unknown>;
}

export type ServiceEvent =
  | { type: "job"; job: JobView }
  | { type: "server"; state: ServerState; owned: boolean; error: AppErrorBody | null }
  | { type: "gallery"; id: string; change: "added" | "updated" | "deleted" };
