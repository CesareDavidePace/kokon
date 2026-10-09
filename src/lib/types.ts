// Mirrors the kokon-service JSON contract (crates/qls-core).

export type Mode = "auto" | "generate" | "edit" | "inpaint" | "outpaint" | "multi_reference" | "upscale";
export type ResolvedMode = Exclude<Mode, "auto">;
/** What an inpaint does in the painted area. */
export type InpaintAction = "edit" | "remove" | "replace" | "add";
/** Presets of the Turbo model (8 steps, no CFG): ~1 MP, ~1.5K, native 2K. */
export type Quality = "fast" | "balanced" | "quality";
export type ResolutionMode = "auto" | "source" | "1k" | "1.5k" | "2k" | "custom";
export type OutputFormat = "png" | "jpeg" | "webp";
export type Units = "pixels" | "percent";
/** Refine effort of older versions' two-pass records (gallery details only). */
export type RefineMode = "off" | "fast" | "best";
/** The upscaler model only (older versions also had "detailed"). */
export type UpscaleMethod = "quick";

export interface UpscaleSpec { method: UpscaleMethod; scale?: number | null; width?: number | null; height?: number | null }

export interface Size { width: number; height: number }
export interface Rect { x: number; y: number; width: number; height: number }

export interface OutpaintSpec { left: number; right: number; top: number; bottom: number; units: Units }

export interface Advanced {
  sampler?: string | null;
  strength?: number | null;
  batch_count?: number | null;
  output_format?: OutputFormat | null;
  vae_tiling?: boolean | null;
  /** Keep a HEIC source's HDR gain map (Ultra HDR JPEG); null = on. */
  hdr?: boolean | null;
}

export interface GenerationRequest {
  prompt: string;
  negative_prompt?: string | null;
  mode: Mode;
  images: string[];
  mask?: string | null;
  inpaint_action?: InpaintAction;
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
  /** Private mode: kept out of the gallery and the output folder, deleted when it ends. */
  private?: boolean;
}

export interface WorkingCanvas { source: Size; content: Size; canvas: Size; content_offset: [number, number]; scale: number }
export interface RefPlan { index: number; source: Size; canvas: WorkingCanvas }

export interface Plan {
  mode: ResolvedMode;
  /** Inpaint action actually run (a "remove ..." prompt makes an edit a removal). */
  inpaint_action?: InpaintAction | null;
  auto_mode: boolean;
  mode_reason: string;
  output: Size;
  working: Size;
  content: Rect;
  steps: number;
  cfg: number;
  sampler: string;
  strength: number | null;
  batch_count: number;
  vae_tiling: boolean;
  references: RefPlan[];
  aspect_label: string;
  notes: string[];
  optimized_for_vram: boolean;
  /** Calibrated to this machine after its first image; 0 = not measured yet. */
  estimated_seconds: number;
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
  /** May hold an older version's preset name ("turbo", "ultra"): see `normalizeQuality`. */
  quality: Quality | string;
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
  gpu: { name: string; vendor: string; unified: boolean; memory_total_mib: number; memory_used_mib: number; memory_free_mib: number; driver_version: string } | null;
  active_job: JobView | null;
  queue: { queued: number; jobs: JobView[] };
  upscaler?: { model: string; available: boolean };
  setup_complete: boolean;
}

export interface Check { id: string; label: string; ok: boolean; detail: string }
export interface SetupReport { ready: boolean; checks: Check[]; devices: [string, string][]; backend_version: string | null }

export type TextEncoderPlacement = "ram" | "gpu" | "cpu" | "auto" | "disk";

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

/** stable-diffusion.cpp build: CUDA (NVIDIA) or Vulkan (AMD, Intel, also NVIDIA). */
export type GpuBackend = "cuda" | "vulkan" | "metal";

export interface AppConfig {
  schemaVersion: number;
  sdcppDir: string;
  gpuBackend: GpuBackend;
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
  extraServerArgs: string[];
  outputDir: string;
  keepServerRunning: boolean;
  idleUnloadMinutes: number;
  setupComplete: boolean;
  /** Hardware preset chosen in the setup wizard. */
  hardwarePreset: string;
  /** Plan as if the GPU had this much VRAM (MiB); 0 = detected. */
  vramOverrideMib: number;
  ui: Record<string, unknown>;
}

export type InstallItemState = "present" | "pending" | "downloading" | "verifying" | "extracting" | "done" | "failed" | "cancelled";

export interface InstallItem {
  id: string;
  label: string;
  kind: "engine" | "model" | "optional";
  size: number;
  received: number;
  state: InstallItemState;
  error: string | null;
  hint: string | null;
  url: string;
  page: string;
  path: string;
}

export interface InstallStatus {
  running: boolean;
  complete: boolean;
  preset: string | null;
  sdcpp_dir: string | null;
  items: InstallItem[];
  bytes_per_sec: number;
}

export interface HardwarePreset {
  id: string;
  label: string;
  summary: string;
  verified: boolean;
  min_vram_mib: number;
  files: InstallItem[];
  download_bytes: number;
  disk_bytes: number;
}

export interface HardwareReport {
  gpu: StatusResponse["gpu"];
  nvidia: boolean;
  vendor: string | null;
  /** Engine the presets' file lists are for. */
  backend: GpuBackend;
  recommended_backend: GpuBackend;
  /** Engine in use now. */
  current_backend: GpuBackend;
  vram_mib: number | null;
  vram_simulated: boolean;
  ram_total_mib: number;
  recommended: string;
  current: string;
  presets: HardwarePreset[];
  sdcpp_dir: string;
  default_sdcpp_dir: string;
  free_bytes: number | null;
  issues_url: string;
}

export type ServiceEvent =
  | { type: "job"; job: JobView }
  | { type: "server"; state: ServerState; owned: boolean; error: AppErrorBody | null }
  | { type: "gallery"; id: string; change: "added" | "updated" | "deleted" };
