//! The application engine shared by every front end (desktop UI, HTTP API, MCP):
//! a queue of generation jobs executed one at a time against sd-server, with truthful
//! progress, cancellation, automatic VRAM mitigation, and gallery persistence.

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use image::GrayImage;
use serde::Serialize;
use tokio::sync::{Notify, broadcast};

use crate::config::AppConfig;
use crate::diagnostics::{GpuInfo, query_gpus};
use crate::error::{AppError, AppResult, ErrorCode, translate_generation_failure};
use crate::geometry::Size;
use crate::imaging::{self, LoadedImage};
use crate::paths;
use crate::pipeline;
use crate::planner::{self, GenerationRequest, Mode, OutputFormat, Plan, PlanInputs, ResolutionMode, ResolvedMode, UpscaleMethod};
use crate::process::{BackendProgress, BackendStage, ProgressParser, ServerManager, ServerState};
use crate::sdcpp::{BackendJobStatus, CancelOutcome, SdcppClient, UpscaleRequest};
use crate::store::{GenerationRecord, InputRecord, PipelineRecord, ReferenceRecord, Store, new_id, now_rfc3339};

pub const APP_VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Queued,
    StartingBackend,
    Preparing,
    Running,
    Saving,
    Done,
    Failed,
    Cancelled,
}

impl JobStatus {
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Done | Self::Failed | Self::Cancelled)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct JobView {
    pub id: String,
    pub status: JobStatus,
    /// Backend-reported stage while running (None if the backend gives no detail).
    pub stage: Option<BackendStage>,
    pub step: Option<(u32, u32)>,
    pub image: Option<(u32, u32)>,
    pub seconds_per_step: Option<f32>,
    pub queue_position: usize,
    pub created_at: String,
    pub started_at: Option<String>,
    pub finished_at: Option<String>,
    pub prompt: String,
    pub client: Option<String>,
    pub plan: Option<Plan>,
    pub seed: Option<i64>,
    pub results: Vec<GenerationRecord>,
    pub error: Option<AppError>,
    pub notes: Vec<String>,
    /// True when progress comes from the server log (we launched it).
    pub detailed_progress: bool,
}

struct Job {
    view: JobView,
    request: GenerationRequest,
    cancel: bool,
    retried_oom: bool,
    workspace: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    Job { job: Box<JobView> },
    Server { state: ServerState, owned: bool, error: Option<AppError> },
    Gallery { id: String, change: &'static str },
}

pub struct Engine {
    cfg: RwLock<AppConfig>,
    pub server: Arc<ServerManager>,
    pub store: Arc<Store>,
    /// Private mode: records in memory, files under `private_dir`; wiped by `clear_private`,
    /// at start and at exit. Never listed in the gallery.
    pub private_store: Arc<Store>,
    /// Measured speed of this machine, for time estimates.
    speed: crate::speed::SpeedBook,
    private_dir: PathBuf,
    jobs: Mutex<HashMap<String, Job>>,
    order: Mutex<VecDeque<String>>,
    active: Mutex<Option<String>>,
    wake: Notify,
    pub events: broadcast::Sender<Event>,
    gpu: Mutex<(Option<Instant>, Option<GpuInfo>)>,
    /// First-run downloads (setup wizard).
    pub installer: Arc<crate::installer::Installer>,
    backend_version: Mutex<Option<String>>,
    last_activity: Mutex<Instant>,
}

fn random_seed() -> i64 {
    let b = uuid::Uuid::new_v4();
    let v = u32::from_le_bytes(b.as_bytes()[..4].try_into().unwrap());
    (v & 0x7fff_ffff) as i64
}

/// Runs CPU-heavy image work off the async runtime.
async fn blocking<T: Send + 'static>(f: impl FnOnce() -> AppResult<T> + Send + 'static) -> AppResult<T> {
    tokio::task::spawn_blocking(f).await.map_err(AppError::internal)?
}

/// Planner inputs from the configuration and measured facts.
pub fn plan_inputs(cfg: &AppConfig, image_sizes: Vec<Size>, mask: Option<(Size, Option<crate::geometry::Rect>, f64)>, vram_total_mib: Option<u64>) -> PlanInputs {
    PlanInputs {
        image_sizes,
        mask,
        vram_total_mib,
        upscaler: cfg.upscaler_file().map(|_| cfg.upscaler_model.trim().to_string()),
        refine: cfg.refine_mode,
        server_vae_tiling: cfg.vae_tile_size > 0,
        sage_attention: cfg.uses_sage_attention(),
        speed_factor: None,
        hide_uncalibrated: !crate::speed::trust_uncalibrated(cfg),
    }
}

impl Engine {
    pub fn new(cfg: AppConfig, data_dir: &Path) -> AppResult<Arc<Self>> {
        let store = Arc::new(Store::open(data_dir)?);
        let private_dir = data_dir.join("private");
        // Leftovers of a crash or forced exit while private mode was on.
        let _ = std::fs::remove_dir_all(&private_dir);
        for sub in ["inputs", "thumbs", "out"] {
            std::fs::create_dir_all(private_dir.join(sub)).map_err(|e| AppError::io("create the data folder", &private_dir, e))?;
        }
        let private_store = Arc::new(Store::open_in_memory(&private_dir)?);
        let (events, _) = broadcast::channel(512);
        Ok(Arc::new(Self {
            cfg: RwLock::new(cfg),
            server: ServerManager::new(data_dir.to_path_buf()),
            store,
            private_store,
            speed: if cfg!(test) { crate::speed::SpeedBook::in_memory() } else { crate::speed::SpeedBook::open(data_dir) },
            private_dir,
            jobs: Mutex::new(HashMap::new()),
            order: Mutex::new(VecDeque::new()),
            active: Mutex::new(None),
            wake: Notify::new(),
            events,
            gpu: Mutex::new((None, None)),
            installer: crate::installer::Installer::new(),
            backend_version: Mutex::new(None),
            last_activity: Mutex::new(Instant::now()),
        }))
    }

    pub fn config(&self) -> AppConfig {
        self.cfg.read().unwrap().clone()
    }

    /// Replaces the configuration. Returns true when the backend must restart.
    pub fn set_config(&self, new: AppConfig) -> AppResult<bool> {
        new.validate()?;
        new.save()?;
        let mut g = self.cfg.write().unwrap();
        let restart = crate::process::build_server_args(&g, g.profile()) != crate::process::build_server_args(&new, new.profile())
            || g.sdcpp_dir != new.sdcpp_dir
            || g.gpu_backend != new.gpu_backend;
        *g = new;
        *self.backend_version.lock().unwrap() = None;
        Ok(restart)
    }

    /// The store holding generation `id`: the gallery, or the private-mode store.
    pub fn gen_store(&self, id: &str) -> &Arc<Store> {
        if self.store.get_generation(id).is_err() && self.private_store.get_generation(id).is_ok() { &self.private_store } else { &self.store }
    }

    /// The store holding input `id` (see [`Self::gen_store`]).
    pub fn input_store(&self, id: &str) -> &Arc<Store> {
        if self.store.get_input(id).is_err() && self.private_store.get_input(id).is_ok() { &self.private_store } else { &self.store }
    }

    /// Where a generation's image file may be deleted from: the output folder, or the
    /// private folder for private results.
    pub fn output_root(&self, id: &str) -> PathBuf {
        if Arc::ptr_eq(self.gen_store(id), &self.private_store) { self.private_dir.join("out") } else { self.config().output_dir }
    }

    /// Ends private mode: forgets every private record and deletes its files.
    pub fn clear_private(&self) -> AppResult<usize> {
        let (items, _) = self.private_store.list_generations(&crate::store::ListQuery { limit: Some(500), ..Default::default() })?;
        self.private_store.clear()?;
        let mut removed = 0;
        for sub in ["inputs", "thumbs", "out"] {
            let dir = self.private_dir.join(sub);
            if let Ok(entries) = std::fs::read_dir(&dir) {
                for e in entries.flatten() {
                    if std::fs::remove_file(e.path()).is_ok() {
                        removed += 1;
                    }
                }
            }
        }
        if !items.is_empty() {
            tracing::info!("private mode cleared ({} results, {removed} files)", items.len());
        }
        Ok(items.len())
    }

    pub fn touch(&self) {
        *self.last_activity.lock().unwrap() = Instant::now();
    }

    pub fn idle_for(&self) -> Duration {
        self.last_activity.lock().unwrap().elapsed()
    }

    pub fn has_pending_work(&self) -> bool {
        self.active.lock().unwrap().is_some() || self.jobs.lock().unwrap().values().any(|j| !j.view.status.is_terminal()) || self.installer.is_running()
    }

    pub fn gpu(&self) -> Option<GpuInfo> {
        let mut g = self.gpu.lock().unwrap();
        if g.0.map(|t| t.elapsed() < Duration::from_secs(3)).unwrap_or(false) {
            return g.1.clone();
        }
        let info = query_gpus().and_then(|v| v.into_iter().next());
        *g = (Some(Instant::now()), info.clone());
        info
    }

    pub fn backend_version(&self) -> Option<String> {
        if let Some(v) = self.backend_version.lock().unwrap().clone() {
            return Some(v);
        }
        let v = crate::diagnostics::backend_version(&self.config().server_exe());
        *self.backend_version.lock().unwrap() = v.clone();
        v
    }

    fn emit_job(&self, id: &str) {
        if let Some(v) = self.job(id) {
            let _ = self.events.send(Event::Job { job: Box::new(v) });
        }
    }

    pub fn emit_server(&self) {
        let s = self.server.status();
        let _ = self.events.send(Event::Server { state: s.state, owned: s.owned, error: s.error });
    }

    fn update(&self, id: &str, f: impl FnOnce(&mut Job)) {
        {
            let mut jobs = self.jobs.lock().unwrap();
            if let Some(j) = jobs.get_mut(id) {
                f(j);
            }
        }
        self.emit_job(id);
    }

    fn queue_positions(&self) {
        let order = self.order.lock().unwrap();
        let mut jobs = self.jobs.lock().unwrap();
        let mut pos = 0;
        for id in order.iter() {
            if let Some(j) = jobs.get_mut(id)
                && j.view.status == JobStatus::Queued
            {
                pos += 1;
                j.view.queue_position = pos;
            }
        }
    }

    pub fn job(&self, id: &str) -> Option<JobView> {
        self.jobs.lock().unwrap().get(id).map(|j| j.view.clone())
    }

    pub fn jobs(&self) -> Vec<JobView> {
        let order = self.order.lock().unwrap();
        let jobs = self.jobs.lock().unwrap();
        order.iter().filter_map(|id| jobs.get(id).map(|j| j.view.clone())).collect()
    }

    pub fn active_job(&self) -> Option<JobView> {
        let id = self.active.lock().unwrap().clone()?;
        self.job(&id)
    }

    /// Validates cheaply and enqueues. `workspace` is the caller's project folder (MCP),
    /// used to resolve relative paths and as an allowed output root.
    pub fn submit(&self, mut request: GenerationRequest, workspace: Option<PathBuf>) -> AppResult<JobView> {
        self.touch();
        let upscale = request.mode == Mode::Upscale;
        // Removing a painted area needs no description (the planner writes the prompt).
        let removal = request.mask.is_some() && request.inpaint_action == planner::InpaintAction::Remove;
        if request.prompt.trim().is_empty() && !upscale && !removal {
            return Err(AppError::invalid("Describe what to generate or change."));
        }
        if upscale && request.images.is_empty() {
            return Err(AppError::invalid("Choose an image to upscale."));
        }
        if request.images.len() > 16 {
            return Err(AppError::invalid("Too many images."));
        }
        if let Some(seed) = request.seed
            && seed < 0
        {
            request.seed = None;
        }
        let id = format!("job_{}", &uuid::Uuid::new_v4().simple().to_string()[..12]);
        let view = JobView {
            id: id.clone(),
            status: JobStatus::Queued,
            stage: None,
            step: None,
            image: None,
            seconds_per_step: None,
            queue_position: 0,
            created_at: now_rfc3339(),
            started_at: None,
            finished_at: None,
            prompt: if upscale && request.prompt.trim().is_empty() {
                match request.upscale.map(|u| u.method).unwrap_or_default() {
                    UpscaleMethod::Quick => "Upscale (Quick)".into(),
                    UpscaleMethod::Detailed => "Upscale (Detailed)".into(),
                }
            } else {
                request.prompt.clone()
            },
            client: request.client.clone(),
            plan: None,
            seed: request.seed,
            results: vec![],
            error: None,
            notes: vec![],
            detailed_progress: false,
        };
        {
            let mut jobs = self.jobs.lock().unwrap();
            jobs.insert(id.clone(), Job { view, request, cancel: false, retried_oom: false, workspace });
            let mut order = self.order.lock().unwrap();
            order.push_back(id.clone());
            // Keep memory bounded: drop the oldest finished jobs.
            while order.len() > 300 {
                let Some(pos) = order.iter().position(|i| jobs.get(i).map(|j| j.view.status.is_terminal()).unwrap_or(true)) else { break };
                if let Some(old) = order.remove(pos) {
                    jobs.remove(&old);
                }
            }
        }
        self.queue_positions();
        self.emit_job(&id);
        self.wake.notify_one();
        Ok(self.job(&id).unwrap())
    }

    /// Requests cancellation. Queued jobs stop immediately; a running job is stopped as
    /// soon as the backend allows (see `run_job`).
    pub fn cancel(&self, id: &str) -> AppResult<JobView> {
        let status = self.job(id).ok_or_else(|| AppError::new(ErrorCode::NotFound, "Unknown job."))?.status;
        if status.is_terminal() {
            return Ok(self.job(id).unwrap());
        }
        self.update(id, |j| {
            j.cancel = true;
            if j.view.status == JobStatus::Queued {
                j.view.status = JobStatus::Cancelled;
                j.view.finished_at = Some(now_rfc3339());
            }
        });
        self.queue_positions();
        Ok(self.job(id).unwrap())
    }

    /// Cancels every unfinished job submitted by `client` (e.g. "ui" when the desktop app
    /// closes). A running job stops the owned server, which frees the GPU right away.
    pub fn cancel_client_jobs(&self, client: &str) -> usize {
        let ids: Vec<String> = {
            let jobs = self.jobs.lock().unwrap();
            jobs.values().filter(|j| !j.view.status.is_terminal() && j.view.client.as_deref() == Some(client)).map(|j| j.view.id.clone()).collect()
        };
        for id in &ids {
            let _ = self.cancel(id);
        }
        ids.len()
    }

    fn next_queued(&self) -> Option<String> {
        let order = self.order.lock().unwrap();
        let jobs = self.jobs.lock().unwrap();
        order.iter().find(|id| jobs.get(*id).map(|j| j.view.status == JobStatus::Queued).unwrap_or(false)).cloned()
    }

    /// Worker loop: one job at a time (the GPU is the bottleneck); the queue keeps order.
    pub async fn run_worker(self: Arc<Self>) {
        loop {
            let Some(id) = self.next_queued() else {
                self.wake.notified().await;
                continue;
            };
            *self.active.lock().unwrap() = Some(id.clone());
            self.queue_positions();
            let r = self.clone().run_job(&id).await;
            if let Err(e) = r {
                let cancelled = self.jobs.lock().unwrap().get(&id).map(|j| j.cancel).unwrap_or(false);
                self.update(&id, |j| {
                    if cancelled || e.code == ErrorCode::Cancelled {
                        j.view.status = JobStatus::Cancelled;
                    } else {
                        j.view.status = JobStatus::Failed;
                        j.view.error = Some(e.clone());
                    }
                    j.view.finished_at = Some(now_rfc3339());
                    j.view.stage = None;
                });
                if !cancelled {
                    tracing::warn!("job {id} failed: {} {:?}", e.message, e.detail);
                }
            }
            *self.active.lock().unwrap() = None;
            self.touch();
        }
    }

    fn is_cancelled(&self, id: &str) -> bool {
        self.jobs.lock().unwrap().get(id).map(|j| j.cancel).unwrap_or(true)
    }

    fn check_cancel(&self, id: &str) -> AppResult<()> {
        if self.is_cancelled(id) { Err(AppError::new(ErrorCode::Cancelled, "Cancelled.")) } else { Ok(()) }
    }

    /// Loads one image source: "input:<id>", "gen:<id>" or an absolute path. A private job
    /// keeps its copy of the input in the private store.
    pub fn resolve_source(&self, src: &str, workspace: Option<&Path>, private: bool) -> AppResult<(LoadedImage, InputRecord)> {
        let (bytes, original_path, name) = if let Some(id) = src.strip_prefix("input:") {
            let rec = self.input_store(id).get_input(id)?;
            let b = std::fs::read(&rec.path).map_err(|e| AppError::io("read the stored image", Path::new(&rec.path), e))?;
            (b, rec.original_path.clone(), rec.original_name.clone())
        } else if let Some(id) = src.strip_prefix("gen:") {
            let rec = self.gen_store(id).get_generation(id)?;
            let p = PathBuf::from(&rec.file_path);
            let b = std::fs::read(&p).map_err(|e| AppError::io("read the generated image", &p, e))?;
            (b, Some(rec.file_path.clone()), p.file_name().map(|n| n.to_string_lossy().into_owned()))
        } else {
            let p = paths::validate_input_image(src, workspace, &[])?;
            let b = std::fs::read(&p).map_err(|e| AppError::io("read the image", &p, e))?;
            (b, Some(p.to_string_lossy().into_owned()), p.file_name().map(|n| n.to_string_lossy().into_owned()))
        };
        let store = if private { &self.private_store } else { &self.store };
        let rec = store.put_input(&bytes, name.as_deref(), original_path.as_deref())?;
        let img = imaging::decode_image(&bytes)?;
        Ok((img, rec))
    }

    /// How this machine's measured speed compares with the time model (None: not measured).
    fn speed_factor(&self, cfg: &AppConfig) -> Option<f32> {
        self.speed.get(&crate::speed::machine_key(cfg, self.gpu().as_ref())).map(|c| c.factor)
    }

    /// Dry run: the plan a request would get (used by the UI to show working sizes).
    pub fn preview_plan(&self, req: &GenerationRequest, sizes: Vec<Size>, mask: Option<(Size, Option<crate::geometry::Rect>, f64)>) -> AppResult<Plan> {
        let cfg = self.config();
        let vram = cfg.planning_vram_mib(self.gpu().map(|g| g.planning_mib()));
        let mut inputs = plan_inputs(&cfg, sizes, mask, vram);
        inputs.speed_factor = self.speed_factor(&cfg);
        planner::plan(req, &inputs, &cfg.profile().recommended_settings)
    }

    async fn run_job(self: Arc<Self>, id: &str) -> AppResult<()> {
        let started = Instant::now();
        let (request, workspace) = {
            let jobs = self.jobs.lock().unwrap();
            let j = jobs.get(id).ok_or_else(|| AppError::internal("job vanished"))?;
            (j.request.clone(), j.workspace.clone())
        };
        self.update(id, |j| {
            j.view.status = JobStatus::Preparing;
            j.view.started_at = Some(now_rfc3339());
            j.view.queue_position = 0;
        });

        // 1. Load inputs (off the async runtime: decoding large photos is CPU work).
        let this = self.clone();
        let req2 = request.clone();
        let ws = workspace.clone();
        let (images, inputs_meta, mask) = tokio::task::spawn_blocking(move || -> AppResult<_> {
            let mut images = Vec::new();
            let mut metas = Vec::new();
            for s in &req2.images {
                let (img, rec) = this.resolve_source(s, ws.as_deref(), req2.private)?;
                images.push(img);
                metas.push(rec);
            }
            let mask = match &req2.mask {
                Some(m) => {
                    let (img, rec) = this.resolve_source(m, ws.as_deref(), req2.private)?;
                    Some((imaging::mask_from_image(&img), rec))
                }
                None => None,
            };
            Ok((images, metas, mask))
        })
        .await
        .map_err(AppError::internal)??;
        self.check_cancel(id)?;

        // 2. Plan.
        let cfg = self.config();
        let profile = cfg.profile().clone();
        let vram = cfg.planning_vram_mib(self.gpu().map(|g| g.planning_mib()));
        let mask_info = mask.as_ref().map(|(m, _)| (Size::new(m.width(), m.height()), imaging::mask_bbox(m, 16), imaging::mask_coverage(m, 16)));
        let mut plan_inputs = plan_inputs(&cfg, images.iter().map(|i| i.size()).collect(), mask_info, vram);
        plan_inputs.speed_factor = self.speed_factor(&cfg);
        let plan = planner::plan(&request, &plan_inputs, &profile.recommended_settings)?;
        let seed = request.seed.unwrap_or_else(random_seed);
        let mut request = request;
        if plan.mode == ResolvedMode::Upscale && request.prompt.trim().is_empty() {
            request.prompt = self.upscale_prompt(&request, plan.steps == 0);
        }
        self.update(id, |j| {
            j.view.plan = Some(plan.clone());
            j.view.seed = Some(seed);
            j.view.notes = plan.notes.clone();
        });

        // 3. Backend.
        // A job that also starts the server is not a fair speed sample (startup, first load).
        let mut cold_start = false;
        if self.server.status().state != ServerState::Ready || self.server.owned_alive() == Some(false) {
            cold_start = true;
            self.update(id, |j| j.view.status = JobStatus::StartingBackend);
            self.emit_server();
            let r = self.server.ensure_ready(&cfg).await;
            self.emit_server();
            r?;
        }
        self.check_cancel(id)?;
        self.update(id, |j| j.view.status = JobStatus::Preparing);

        // Which upscalers the running server really has (an external server may lack
        // --hires-upscalers-dir): enlargements fall back to Lanczos.
        // Fresh, because the server rescans the upscalers folder on every request.
        let client = SdcppClient::new(cfg.server_port);
        let caps = client.capabilities(Duration::from_secs(5)).await.ok().or_else(|| self.server.capabilities()).unwrap_or_default();
        let esrgan = plan.upscaler.clone().filter(|n| caps.has_upscaler(n, true));
        let needs_esrgan = plan.upscale.is_some() || plan.output.area() as f64 > plan.content.area() as f64 * 1.21;
        if needs_esrgan && plan.upscaler.is_some() && esrgan.is_none() {
            let name = plan.upscaler.clone().unwrap_or_default();
            self.update(id, |j| j.view.notes.push(format!("The inference server did not list the upscaler model {name}; Lanczos was used instead.")));
        } else if plan.upscaler.is_none() && plan.upscale.is_some() {
            self.update(id, |j| {
                j.view.notes.push(format!("Upscaler model {} not found; enlarged with Lanczos. Put {}.pth in the upscalers folder for sharper results.", cfg.upscaler_model, cfg.upscaler_model))
            });
        }
        let quick_upscale = plan.upscale.map(|u| u.method == UpscaleMethod::Quick).unwrap_or(false);
        let icc = images.first().and_then(|i| i.icc_profile.clone());

        let (contents, post) = if quick_upscale {
            // No diffusion: the source itself is enlarged.
            let src = images.into_iter().next().ok_or_else(|| AppError::invalid("Choose an image to upscale."))?;
            let size = src.size();
            let post = pipeline::PostProcess {
                kind: pipeline::PostKind::Plain,
                keep_alpha: true,
                working: size,
                content: crate::geometry::Rect::new(0, 0, size.width, size.height),
                output: plan.output,
                mode: plan.mode,
                outpaint: None,
            };
            (vec![src.pixels], post)
        } else {
            let mut images = images;
            if plan.upscale.is_some() {
                // Detailed: bring the source to the refine size first (ESRGAN when enlarging).
                let target = Size::new(plan.content.width, plan.content.height);
                let src = std::mem::replace(&mut images[0].pixels, image::RgbaImage::new(1, 1));
                let (enlarged, _) = self.enlarge(id, &client, src, Some(target), esrgan.as_deref()).await?;
                images[0].pixels = enlarged;
                self.update(id, |j| {
                    j.view.status = JobStatus::Preparing;
                    j.view.stage = None;
                });
            }

            // 4. Build the request (CPU work).
            let plan2 = plan.clone();
            let prompt = request.prompt.clone();
            let mask_img: Option<GrayImage> = mask.as_ref().map(|(m, _)| m.clone());
            let opts = pipeline::BackendOptions { server_vae_tile_size: cfg.vae_tile_size };
            let prepared = tokio::task::spawn_blocking(move || pipeline::prepare(&plan2, &prompt, seed, &images, mask_img.as_ref(), &opts))
                .await
                .map_err(AppError::internal)??;
            let pipeline::PreparedJob { request: sd_req, post } = prepared;

            // 5. Submit and follow.
            let mut lines = self.server.subscribe_lines();
            let detailed = self.server.status().owned;
            let submitted = Instant::now();
            let accepted = client.submit(&sd_req).await?;
            drop(sd_req); // release the base64 payloads
            self.update(id, |j| {
                j.view.status = JobStatus::Running;
                j.view.detailed_progress = detailed;
            });
            let mut parser = ProgressParser::default();
            let mut job_log: Vec<String> = Vec::new();
            let mut last_poll = Instant::now() - Duration::from_secs(10);
            let backend_job = loop {
                // Drain log lines -> truthful stage/step.
                loop {
                    match lines.try_recv() {
                        Ok(l) => {
                            if job_log.len() < 4000 {
                                job_log.push(l.clone());
                            }
                            if let Some(p) = parser.feed(&l) {
                                self.apply_progress(id, p);
                            }
                        }
                        Err(broadcast::error::TryRecvError::Lagged(_)) => continue,
                        Err(_) => break,
                    }
                }
                if self.is_cancelled(id) {
                    return self.cancel_running(id, &client, &accepted.id, &cfg).await;
                }
                if last_poll.elapsed() >= Duration::from_millis(500) {
                    last_poll = Instant::now();
                    match client.job(&accepted.id).await {
                        Ok(bj) => match bj.status {
                            BackendJobStatus::Completed | BackendJobStatus::Failed | BackendJobStatus::Cancelled => break bj,
                            BackendJobStatus::Queued if !detailed => self.update(id, |j| j.view.stage = None),
                            BackendJobStatus::Generating if !detailed => self.update(id, |j| j.view.stage = Some(BackendStage::Generating)),
                            _ => {}
                        },
                        Err(e) => {
                            if self.server.owned_alive() == Some(false) {
                                let tail = self.server.log.tail(30).join("\n");
                                self.server.mark_lost(AppError::new(ErrorCode::ServerCrashed, "The inference server stopped unexpectedly."));
                                self.emit_server();
                                let mut err = translate_generation_failure("server process exited", &tail, (plan.working.width, plan.working.height), None);
                                if err.code == ErrorCode::GenerationFailed {
                                    err = AppError::new(ErrorCode::ServerCrashed, "The inference server stopped unexpectedly.")
                                        .suggest("It will restart automatically on the next job.")
                                        .detail(tail);
                                }
                                return self.maybe_retry_oom(id, err).await;
                            }
                            if e.code == ErrorCode::NotFound {
                                return Err(AppError::new(ErrorCode::GenerationFailed, "The inference server lost track of the job (it may have restarted)."));
                            }
                        }
                    }
                }
                tokio::time::sleep(Duration::from_millis(120)).await;
            };

            match backend_job.status {
                BackendJobStatus::Completed => {}
                BackendJobStatus::Cancelled => return Err(AppError::new(ErrorCode::Cancelled, "Cancelled.")),
                _ => {
                    let msg = backend_job.error.map(|e| format!("{}: {}", e.code, e.message)).unwrap_or_default();
                    let smaller = Size::new(((plan.working.width as f64) * 0.75) as u32 / 32 * 32, ((plan.working.height as f64) * 0.75) as u32 / 32 * 32);
                    let err = translate_generation_failure(&msg, &job_log.join("\n"), (plan.working.width, plan.working.height), Some((smaller.width, smaller.height)));
                    return self.maybe_retry_oom(id, err).await;
                }
            }
            if !cold_start {
                let key = crate::speed::machine_key(&cfg, self.gpu().as_ref());
                let predicted = planner::backend_seconds(&plan, cfg.uses_sage_attention());
                if let Some(f) = self.speed.record(&key, submitted.elapsed().as_secs_f32(), predicted) {
                    tracing::info!("speed calibration {key}: x{f:.2}");
                }
            }
            let result = backend_job.result.unwrap_or_default();
            if result.images.is_empty() {
                return Err(AppError::new(ErrorCode::GenerationFailed, "The inference server returned no image."));
            }
            let post = Arc::new(post);
            let mut contents = Vec::with_capacity(result.images.len());
            for im in result.images {
                let p = post.clone();
                contents.push(
                    blocking(move || {
                        let raw = imaging::decode_image(&crate::sdcpp::b64_decode(&im.b64_json)?)?.pixels;
                        Ok(pipeline::crop_content(raw, &p))
                    })
                    .await?,
                );
            }
            (contents, Arc::try_unwrap(post).map_err(|_| AppError::internal("post-process still shared"))?)
        };

        // 6. Enlarge (ESRGAN), post-process and save.
        let post = Arc::new(post);
        let duration_start = started;
        let mask_rec = mask.map(|(_, r)| r);
        let mut records = Vec::with_capacity(contents.len());
        for (i, content) in contents.into_iter().enumerate() {
            let before = Size::new(content.width(), content.height());
            let (content, enlarged_with) = self.enlarge(id, &client, content, pipeline::enlarge_target(&post), esrgan.as_deref()).await?;
            self.update(id, |j| {
                j.view.status = JobStatus::Saving;
                j.view.stage = None;
            });
            let pipeline_rec = PipelineRecord {
                base: None,
                refine: None,
                refine_steps: None,
                refine_denoise: None,
                hires_upscaler: None,
                cache_threshold: plan.cache_threshold,
                enlarged_from: enlarged_with.is_some().then_some(before),
                enlarged_with,
            };
            let this = self.clone();
            let mut plan3 = plan.clone();
            // Notes added while running (fallbacks, retries) belong in the record too.
            if let Some(j) = self.job(id) {
                plan3.notes = j.notes;
            }
            let (p, req3, inputs3, mask3, icc3, ws) = (post.clone(), request.clone(), inputs_meta.clone(), mask_rec.clone(), icc.clone(), workspace.clone());
            let rec = blocking(move || {
                let final_img = pipeline::finish(content, &p)?;
                let duration_ms = duration_start.elapsed().as_millis() as u64;
                let pipeline = (pipeline_rec != PipelineRecord::default()).then_some(pipeline_rec);
                this.save_result(&final_img, &req3, &plan3, seed + i as i64, &inputs3, mask3.as_ref(), icc3.as_deref(), duration_ms, ws.as_deref(), i, pipeline)
            })
            .await?;
            if !request.private {
                let _ = self.events.send(Event::Gallery { id: rec.id.clone(), change: "added" });
            }
            records.push(rec);
        }
        self.update(id, |j| {
            j.view.status = JobStatus::Done;
            j.view.results = records;
            j.view.finished_at = Some(now_rfc3339());
            j.view.stage = None;
        });
        Ok(())
    }

    /// Prompt for an upscale request that has none: the source's own prompt for
    /// text-to-image results, otherwise a neutral description (an edit instruction such
    /// as "make it darker" would keep pushing the image during the refine).
    fn upscale_prompt(&self, req: &GenerationRequest, quick: bool) -> String {
        let source = req.images.first().and_then(|s| s.strip_prefix("gen:")).and_then(|g| self.gen_store(g).get_generation(g).ok());
        match source {
            Some(g) if g.mode == "generate" || (quick && !g.prompt.trim().is_empty()) => g.prompt,
            _ if quick => "Upscaled image".into(),
            _ => planner::UPSCALE_PROMPT.into(),
        }
    }

    /// Enlarges `content` with ESRGAN when [`pipeline::finish`] would resize it up to
    /// `target` by more than ~1.1x. Returns the (possibly larger) content and how the
    /// enlargement is done; ESRGAN failures fall back to Lanczos with a note.
    async fn enlarge(&self, id: &str, client: &SdcppClient, content: image::RgbaImage, target: Option<Size>, esrgan: Option<&str>) -> AppResult<(image::RgbaImage, Option<String>)> {
        let Some(target) = target else { return Ok((content, None)) };
        let size = Size::new(content.width(), content.height());
        let factor = (target.width as f64 / size.width as f64).max(target.height as f64 / size.height as f64);
        if factor <= pipeline::ESRGAN_MIN_FACTOR {
            return Ok((content, None));
        }
        let Some(name) = esrgan else { return Ok((content, Some("Lanczos".into()))) };
        self.update(id, |j| {
            j.view.status = JobStatus::Running;
            j.view.stage = Some(BackendStage::Upscaling);
            j.view.step = None;
        });
        let content = Arc::new(content);
        let r = self.esrgan_enlarge(id, client, content.clone(), target, name).await;
        let content = Arc::try_unwrap(content).unwrap_or_else(|a| (*a).clone());
        match r {
            Ok(Some(img)) => Ok((img, Some(name.to_string()))),
            Ok(None) => Ok((content, None)),
            Err(e) if e.code == ErrorCode::Cancelled => Err(e),
            Err(e) => {
                tracing::warn!("ESRGAN upscale failed, using Lanczos: {} {:?}", e.message, e.detail);
                self.update(id, |j| j.view.notes.push(format!("Upscaler model failed ({}); enlarged with Lanczos.", e.message)));
                Ok((content, Some("Lanczos".into())))
            }
        }
    }

    async fn esrgan_enlarge(&self, id: &str, client: &SdcppClient, content: Arc<image::RgbaImage>, target: Size, upscaler: &str) -> AppResult<Option<image::RgbaImage>> {
        // Tiles are sized for an x4 model, the common case (an x2 model just gets smaller results).
        let Some(ep) = pipeline::esrgan_plan(Size::new(content.width(), content.height()), target, 4) else { return Ok(None) };
        let c = content.clone();
        let input = Arc::new(blocking(move || Ok(imaging::resize(&c, ep.input.width, ep.input.height))).await?);
        let mut stitcher = pipeline::Stitcher::new(ep.input);
        for tile in &ep.tiles {
            self.check_cancel(id)?;
            let (inp, t) = (input.clone(), *tile);
            let png = blocking(move || pipeline::tile_png_b64(&inp, &t)).await?;
            let resp = client.upscale(&UpscaleRequest { image: png, upscaler: Some(upscaler.to_string()), tile_size: None, output_format: "png".into() }).await?;
            let up = blocking(move || Ok(imaging::decode_image(&crate::sdcpp::b64_decode(&resp.images[0].b64_json)?)?.pixels)).await?;
            stitcher.add(tile, &up)?;
        }
        Ok(Some(blocking(move || stitcher.finish(&content)).await?))
    }

    fn apply_progress(&self, id: &str, p: BackendProgress) {
        self.update(id, |j| {
            j.view.stage = Some(p.stage);
            j.view.step = p.step;
            j.view.image = p.image;
            j.view.seconds_per_step = p.seconds_per_step;
        });
    }

    async fn cancel_running(&self, id: &str, client: &SdcppClient, backend_id: &str, cfg: &AppConfig) -> AppResult<()> {
        match client.cancel(backend_id).await {
            Ok(CancelOutcome::Cancelled) | Ok(CancelOutcome::AlreadyFinished) | Ok(CancelOutcome::NotFound) => {}
            Ok(CancelOutcome::NotInterruptible) | Err(_) => {
                // This sd-server build cannot interrupt a running job. If we own the
                // process, restart it to free the GPU now; otherwise the result is discarded.
                if self.server.status().owned {
                    self.update(id, |j| j.view.notes.push("Stopping the inference server to cancel immediately.".into()));
                    self.server.stop().await;
                    self.emit_server();
                    let this_cfg = cfg.clone();
                    let srv = self.server.clone();
                    let has_more = self.next_queued().is_some();
                    if has_more {
                        tokio::spawn(async move {
                            let _ = srv.ensure_ready(&this_cfg).await;
                        });
                    }
                } else {
                    self.update(id, |j| j.view.notes.push("The external server will finish this image; the result is discarded.".into()));
                }
            }
        }
        Err(AppError::new(ErrorCode::Cancelled, "Cancelled."))
    }

    /// One automatic retry for out-of-memory: enable VAE tiling, then lower resolution.
    async fn maybe_retry_oom(&self, id: &str, err: AppError) -> AppResult<()> {
        let can_retry = err.code == ErrorCode::OutOfMemory && !self.jobs.lock().unwrap().get(id).map(|j| j.retried_oom).unwrap_or(true);
        if !can_retry {
            return Err(err);
        }
        self.update(id, |j| {
            j.retried_oom = true;
            let tiled = j.view.plan.as_ref().map(|p| p.vae_tiling).unwrap_or(false);
            j.request.advanced.vae_tiling = Some(true);
            if tiled {
                j.request.resolution = match j.request.resolution {
                    ResolutionMode::K2 | ResolutionMode::Source | ResolutionMode::Custom => ResolutionMode::K15,
                    _ => ResolutionMode::K1,
                };
                j.request.width = None;
                j.request.height = None;
                j.view.notes.push("Ran out of GPU memory; retrying at a lower resolution.".into());
            } else {
                j.view.notes.push("Ran out of GPU memory; retrying with memory optimization.".into());
            }
            j.view.status = JobStatus::Queued;
            j.view.stage = None;
            j.view.step = None;
        });
        // Re-queue at the front.
        {
            let mut order = self.order.lock().unwrap();
            if let Some(pos) = order.iter().position(|x| x == id) {
                order.remove(pos);
            }
            order.push_front(id.to_string());
        }
        self.wake.notify_one();
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn save_result(
        &self,
        img: &image::RgbaImage,
        req: &GenerationRequest,
        plan: &Plan,
        seed: i64,
        inputs: &[InputRecord],
        mask: Option<&InputRecord>,
        icc: Option<&[u8]>,
        duration_ms: u64,
        workspace: Option<&Path>,
        index: usize,
        pipeline: Option<PipelineRecord>,
    ) -> AppResult<GenerationRecord> {
        let cfg = self.config();
        let id = new_id();
        let now = chrono::Local::now();
        let store = if req.private { &self.private_store } else { &self.store };
        let dir = if req.private { self.private_dir.join("out") } else { cfg.output_dir.join(now.format("%Y-%m-%d").to_string()) };
        std::fs::create_dir_all(&dir).map_err(|e| AppError::io("create the output folder", &dir, e))?;
        let words: Vec<&str> = req.prompt.split_whitespace().take(6).collect();
        let stem = format!("{}-{}-{seed}", now.format("%H%M%S"), paths::sanitize_file_stem(&words.join(" ")));
        // HDR from a HEIC source forces an Ultra HDR JPEG; otherwise the requested format.
        let (hdr, hdr_note) = hdr_for(img, req, plan, inputs);
        let format = if hdr.is_some() { OutputFormat::Jpeg } else { plan.output_format };
        let file = paths::unique_path(dir.join(format!("{stem}.{}", format.ext())), &[]);

        let mode = match plan.mode {
            ResolvedMode::Generate => "generate",
            ResolvedMode::Edit => "edit",
            ResolvedMode::Inpaint => "inpaint",
            ResolvedMode::Outpaint => "outpaint",
            ResolvedMode::MultiReference => "multi_reference",
            ResolvedMode::Upscale => "upscale",
        };
        let mut rec = GenerationRecord {
            id: id.clone(),
            created_at: now_rfc3339(),
            mode: mode.into(),
            prompt: req.prompt.clone(),
            negative_prompt: plan.negative_prompt.clone(),
            seed,
            width: img.width(),
            height: img.height(),
            working_width: plan.working.width,
            working_height: plan.working.height,
            model: cfg.profile().model_name(),
            sampler: plan.sampler.clone(),
            scheduler: plan.scheduler.clone(),
            steps: plan.steps,
            cfg: plan.cfg,
            strength: plan.strength,
            quality: serde_json::to_value(plan.quality).ok().and_then(|v| v.as_str().map(str::to_owned)).unwrap_or_default(),
            resolution: serde_json::to_value(plan.resolution).ok().and_then(|v| v.as_str().map(str::to_owned)).unwrap_or_default(),
            references: inputs
                .iter()
                .enumerate()
                .map(|(i, r)| ReferenceRecord { index: i, input_id: r.id.clone(), original_path: r.original_path.clone(), width: r.width, height: r.height })
                .collect(),
            mask_input_id: mask.map(|m| m.id.clone()),
            outpaint: plan.outpaint.map(|o| o.expansion),
            file_path: file.to_string_lossy().into_owned(),
            extra_paths: vec![],
            favorite: false,
            parent_id: req.parent_id.clone(),
            duration_ms,
            notes: plan.notes.clone(),
            app_version: APP_VERSION.into(),
            backend_version: self.backend_version.lock().unwrap().clone(),
            client: req.client.clone(),
            request: serde_json::to_value(req).unwrap_or_default(),
            pipeline,
        };
        rec.notes.extend(hdr_note);
        let bytes = match &hdr {
            Some(gm) => crate::hdr::encode_ultra_hdr(img, icc, gm, 95).map_err(|e| AppError::new(ErrorCode::Internal, "Failed to encode the HDR JPEG.").detail(e))?,
            None => encode_output(img, format, &rec, icc)?,
        };
        write_atomic(&file, &bytes)?;

        // Optional copy to a caller-chosen path (MCP / API), validated and never overwriting.
        if let Some(out) = &req.output_path {
            let mut roots = vec![cfg.output_dir.clone()];
            if let Some(w) = workspace {
                roots.push(w.to_path_buf());
            }
            let input_paths: Vec<PathBuf> = inputs.iter().filter_map(|r| r.original_path.as_ref().map(PathBuf::from)).collect();
            let raw = if index == 0 { out.clone() } else { numbered(out, index) };
            let target = paths::resolve_output_path(&raw, workspace, &roots, &input_paths)?;
            if let Some(d) = target.parent() {
                std::fs::create_dir_all(d).map_err(|e| AppError::io("create the output folder", d, e))?;
            }
            let ext = target.extension().map(|e| e.to_string_lossy().to_ascii_lowercase()).unwrap_or_default();
            let fmt = match ext.as_str() {
                "jpg" | "jpeg" => OutputFormat::Jpeg,
                "webp" => OutputFormat::Webp,
                _ => OutputFormat::Png,
            };
            let data = if fmt == format { bytes.clone() } else { encode_output(img, fmt, &rec, icc)? };
            write_atomic(&target, &data)?;
            rec.extra_paths.push(target.to_string_lossy().into_owned());
        }
        drop(bytes);
        let thumb = imaging::thumbnail(img, 512);
        if let Ok(j) = imaging::encode_jpeg(&thumb, 85) {
            let _ = std::fs::write(store.thumb_path(&id), j);
        }
        store.insert_generation(&rec)?;
        Ok(rec)
    }
}

/// The gain map to attach when the source is an HDR HEIC and the user kept HDR on
/// (`advanced.hdr`, default on), plus a note saying what happened. Only for results that
/// keep the source's framing (edit, inpaint, upscale); every failure falls back to SDR.
fn hdr_for(img: &image::RgbaImage, req: &GenerationRequest, plan: &Plan, inputs: &[InputRecord]) -> (Option<crate::hdr::GainMap>, Option<String>) {
    let Some(src) = inputs.first().filter(|r| r.format == "heic" && r.hdr) else { return (None, None) };
    if req.advanced.hdr == Some(false) {
        return (None, None);
    }
    if !matches!(plan.mode, ResolvedMode::Edit | ResolvedMode::Inpaint | ResolvedMode::Upscale) {
        return (None, Some("HDR not kept: only edits, inpainting and upscales keep the source's framing.".into()));
    }
    if !crate::hdr::same_framing((src.width, src.height), img.dimensions()) {
        return (None, Some("HDR not kept: the result's framing differs from the HEIC source.".into()));
    }
    let Some(ffmpeg) = crate::hdr::find_ffmpeg() else {
        return (None, Some("HDR not kept: ffmpeg was not found on PATH (needed to read the HEIC gain map).".into()));
    };
    let attempt = || -> Result<(crate::hdr::GainMap, f32), String> {
        let path = Path::new(&src.path);
        let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
        let info = crate::heif::gain_map(&bytes).ok_or("no ISO 21496-1 gain map")?;
        let gm = crate::hdr::read_heic_gain_map(&ffmpeg, path, &info)?;
        let base = imaging::decode_image(&bytes).map_err(|e| e.message)?.pixels;
        if !crate::hdr::same_framing(base.dimensions(), gm.image.dimensions()) {
            return Err("gain map and image disagree on orientation".into());
        }
        Ok(crate::hdr::adapt_gain_map(&gm, &base, img))
    };
    match attempt() {
        Ok((gm, kept)) => {
            let note = format!("HDR kept from the HEIC source (Ultra HDR JPEG); the HDR boost was kept on {:.0}% of the image and faded where the edit changed brightness.", kept * 100.0);
            (Some(gm), Some(note))
        }
        Err(e) => {
            tracing::warn!("HDR gain map failed: {e}");
            (None, Some(format!("HDR not kept: {e}.")))
        }
    }
}

fn numbered(path: &str, i: usize) -> String {
    let p = Path::new(path);
    let stem = p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "image".into());
    let ext = p.extension().map(|s| format!(".{}", s.to_string_lossy())).unwrap_or_default();
    p.with_file_name(format!("{stem}-{}{ext}", i + 1)).to_string_lossy().into_owned()
}

pub fn encode_output(img: &image::RgbaImage, format: OutputFormat, rec: &GenerationRecord, icc: Option<&[u8]>) -> AppResult<Vec<u8>> {
    match format {
        OutputFormat::Png => {
            let text = rec.png_text();
            let pairs: Vec<(&str, &str)> = text.iter().map(|(k, v)| (*k, v.as_str())).collect();
            imaging::encode_png(img, &pairs, icc, false)
        }
        OutputFormat::Jpeg => imaging::encode_jpeg(img, 95),
        OutputFormat::Webp => imaging::encode_webp_lossless(img),
    }
}

pub fn write_atomic(path: &Path, bytes: &[u8]) -> AppResult<()> {
    let tmp = path.with_extension(format!("{}.part", path.extension().map(|e| e.to_string_lossy().into_owned()).unwrap_or_default()));
    std::fs::write(&tmp, bytes).map_err(|e| AppError::io("write the image", &tmp, e))?;
    std::fs::rename(&tmp, path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        AppError::io("save the image", path, e)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RefineMode;

    /// End to end on a real iPhone HEIC (QLS_TEST_HEIC, ffmpeg on PATH): import, simulated
    /// edit (left half 2 stops darker), HDR decision and Ultra HDR file. QLS_TEST_OUT keeps it.
    #[test]
    fn heic_edit_keeps_hdr() {
        let Ok(p) = std::env::var("QLS_TEST_HEIC") else { return };
        if crate::hdr::find_ffmpeg().is_none() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let store = crate::store::Store::open(dir.path()).unwrap();
        let bytes = std::fs::read(&p).unwrap();
        let input = store.put_input(&bytes, Some("IMG.HEIC"), Some(&p)).unwrap();
        assert_eq!(input.format, "heic");
        assert!(input.hdr);
        assert!(store.input_display_path(&input.id).is_file());

        let mut edited = imaging::decode_image(&bytes).unwrap().pixels;
        let w = edited.width();
        for (x, _, px) in edited.enumerate_pixels_mut() {
            if x < w / 2 {
                for c in 0..3 {
                    px.0[c] /= 3;
                }
            }
        }
        let mut req = GenerationRequest { prompt: "darker".into(), images: vec![format!("input:{}", input.id)], preserve_dimensions: true, ..Default::default() };
        let pi = crate::planner::PlanInputs { image_sizes: vec![Size::new(input.width, input.height)], vram_total_mib: Some(16303), ..Default::default() };
        let plan = crate::planner::plan(&req, &pi, &crate::config::RecommendedSettings::default()).unwrap();
        assert_eq!(plan.mode, ResolvedMode::Edit);

        let (gm, note) = hdr_for(&edited, &req, &plan, std::slice::from_ref(&input));
        let gm = gm.expect("HDR kept");
        let note = note.unwrap();
        assert!(note.contains("HDR kept"), "{note}");
        // Roughly the right half keeps its boost.
        let share: f64 = note.split("kept on ").nth(1).unwrap().split('%').next().unwrap().parse().unwrap();
        assert!((35.0..65.0).contains(&share), "{note}");
        let icc = crate::heif::primary_icc(&bytes);
        let out = crate::hdr::encode_ultra_hdr(&edited, icc.as_deref(), &gm, 95).unwrap();
        if let Ok(d) = std::env::var("QLS_TEST_OUT") {
            std::fs::write(Path::new(&d).join("edit_ultrahdr.jpg"), &out).unwrap();
        }

        req.advanced.hdr = Some(false);
        assert!(hdr_for(&edited, &req, &plan, std::slice::from_ref(&input)).0.is_none(), "toggle off");
        let cropped = image::imageops::crop_imm(&edited, 0, 0, w, w).to_image();
        let (none, why) = hdr_for(&cropped, &GenerationRequest { advanced: Default::default(), ..req.clone() }, &plan, std::slice::from_ref(&input));
        assert!(none.is_none() && why.unwrap().contains("framing"));
    }

    #[test]
    fn numbered_paths() {
        assert_eq!(numbered(r"C:\a\hero.png", 1), r"C:\a\hero-2.png");
        assert_eq!(numbered("x", 2), "x-3");
    }

    #[tokio::test]
    async fn submit_validate_and_cancel_queued() {
        let dir = tempfile::tempdir().unwrap();
        // SAFETY: test-only env override, single-threaded setup.
        unsafe { std::env::set_var("QLS_CONFIG_DIR", dir.path().join("cfg")) };
        let e = Engine::new(AppConfig::default(), dir.path()).unwrap();
        assert!(e.submit(GenerationRequest { prompt: " ".into(), ..Default::default() }, None).is_err());
        // Remove needs no prompt; other inpaint actions do.
        let remove = GenerationRequest { images: vec!["gen:x".into()], mask: Some("input:m".into()), inpaint_action: planner::InpaintAction::Remove, ..Default::default() };
        assert!(e.submit(GenerationRequest { inpaint_action: planner::InpaintAction::Replace, ..remove.clone() }, None).is_err());
        let r = e.submit(remove, None).unwrap();
        e.cancel(&r.id).unwrap();
        let j = e.submit(GenerationRequest { prompt: "a fox".into(), seed: Some(-1), ..Default::default() }, None).unwrap();
        assert_eq!(j.status, JobStatus::Queued);
        assert_eq!(j.seed, None, "negative seed means random");
        let j2 = e.submit(GenerationRequest { prompt: "a cat".into(), ..Default::default() }, None).unwrap();
        assert_eq!(e.job(&j2.id).unwrap().queue_position, 2);
        let c = e.cancel(&j.id).unwrap();
        assert_eq!(c.status, JobStatus::Cancelled);
        assert_eq!(e.job(&j2.id).unwrap().queue_position, 1);
        assert_eq!(e.next_queued().as_deref(), Some(j2.id.as_str()));
        assert!(e.cancel("nope").is_err());
        // Upscales need an image but no prompt; the job gets a readable label.
        let up = GenerationRequest { mode: Mode::Upscale, ..Default::default() };
        assert!(e.submit(up.clone(), None).is_err());
        let j3 = e.submit(GenerationRequest { images: vec!["gen:x".into()], ..up }, None).unwrap();
        assert_eq!(j3.prompt, "Upscale (Quick)");
        // Closing the desktop app cancels only its own jobs.
        let ui = e.submit(GenerationRequest { prompt: "ui job".into(), client: Some("ui".into()), ..Default::default() }, None).unwrap();
        let mcp = e.submit(GenerationRequest { prompt: "mcp job".into(), client: Some("mcp".into()), ..Default::default() }, None).unwrap();
        assert_eq!(e.cancel_client_jobs("ui"), 1);
        assert_eq!(e.job(&ui.id).unwrap().status, JobStatus::Cancelled);
        assert_eq!(e.job(&mcp.id).unwrap().status, JobStatus::Queued);
    }

    #[test]
    fn private_results_stay_out_of_the_gallery() {
        let dir = tempfile::tempdir().unwrap();
        let e = Engine::new(AppConfig::default(), dir.path()).unwrap();
        // A tiny PNG imported in private mode lives in the private store only.
        let png = imaging::encode_png(&image::RgbaImage::from_pixel(8, 8, image::Rgba([200, 10, 10, 255])), &[], None, false).unwrap();
        let input = e.private_store.put_input(&png, Some("secret.png"), None).unwrap();
        assert!(e.store.get_input(&input.id).is_err());
        assert!(Arc::ptr_eq(e.input_store(&input.id), &e.private_store));
        assert!(Path::new(&input.path).starts_with(dir.path().join("private")));
        let (_, rec) = e.resolve_source(&format!("input:{}", input.id), None, true).unwrap();
        assert_eq!(rec.id, input.id);
        assert!(e.store.get_input(&input.id).is_err(), "a private job does not copy its input into the gallery");
        // Ending private mode deletes records and files.
        assert_eq!(e.clear_private().unwrap(), 0);
        assert!(e.private_store.get_input(&input.id).is_err());
        assert!(!Path::new(&input.path).exists());
        // A new engine wipes leftovers of a previous run.
        std::fs::write(dir.path().join("private").join("out").join("left.png"), b"x").unwrap();
        drop(e);
        let _e = Engine::new(AppConfig::default(), dir.path()).unwrap();
        assert!(!dir.path().join("private").join("out").join("left.png").exists());
        let req: GenerationRequest = serde_json::from_str(r#"{"prompt":"x"}"#).unwrap();
        assert!(!req.private, "off unless asked");
    }

    #[test]
    fn plan_inputs_follow_config() {
        let dir = tempfile::tempdir().unwrap();
        let mut cfg = AppConfig::default();
        cfg.upscalers_dir = Some(dir.path().to_path_buf());
        let i = plan_inputs(&cfg, vec![], None, Some(16303));
        assert_eq!(i.upscaler, None, "no model file -> Lanczos");
        assert!(i.server_vae_tiling && i.sage_attention);
        assert_eq!(i.refine, RefineMode::Fast);
        std::fs::write(dir.path().join("4x-UltraSharp.pth"), b"x").unwrap();
        cfg.vae_tile_size = 0;
        cfg.refine_mode = RefineMode::Best;
        let i = plan_inputs(&cfg, vec![], None, None);
        assert_eq!(i.upscaler.as_deref(), Some("4x-UltraSharp"));
        assert!(!i.server_vae_tiling);
        assert_eq!(i.refine, RefineMode::Best);
    }

    #[test]
    fn output_encoding_formats() {
        let img = image::RgbaImage::from_pixel(8, 8, image::Rgba([1, 2, 3, 255]));
        let rec: GenerationRecord = serde_json::from_value(serde_json::json!({
            "id":"x","created_at":"t","mode":"generate","prompt":"p","negative_prompt":"","seed":1,"width":8,"height":8,
            "working_width":32,"working_height":32,"model":"m","sampler":"euler","scheduler":null,"steps":1,"cfg":6.0,"strength":null,
            "quality":"fast","resolution":"auto","references":[],"mask_input_id":null,"outpaint":null,"file_path":"f","favorite":false,
            "parent_id":null,"duration_ms":1,"notes":[],"app_version":"1","backend_version":null,"client":null
        }))
        .unwrap();
        let png = encode_output(&img, OutputFormat::Png, &rec, None).unwrap();
        assert!(imaging::read_png_text(&png).iter().any(|(k, _)| k == "qls:generation"));
        let jpg = encode_output(&img, OutputFormat::Jpeg, &rec, None).unwrap();
        assert_eq!(&jpg[..2], &[0xFF, 0xD8]);
        let webp = encode_output(&img, OutputFormat::Webp, &rec, None).unwrap();
        assert_eq!(&webp[..4], b"RIFF");
    }
}
