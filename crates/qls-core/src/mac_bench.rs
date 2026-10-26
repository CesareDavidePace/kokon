//! Settings → Performance → "Benchmark this Mac": runs `kokon-engine-mac bench` on the active model files and
//! keeps its state for the UI. The bench runs each configuration in a child process of its own and prints
//! `PROGRESS k/N <what>` before each one; the report (markdown) goes to a file in the data folder.
//!
//! While it runs the inference server is stopped and held (see [`crate::process::ServerManager::hold`]): the bench
//! needs the whole GPU and memory, and jobs would only slow it down and spoil its numbers.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use serde::Serialize;
use tokio::io::{AsyncBufReadExt, AsyncRead, BufReader};

use crate::config::{AppConfig, ModelProfile};
use crate::error::{AppError, AppResult, ErrorCode};

/// Lines of the bench's output kept to explain a failure.
pub const LOG_LINES: usize = 200;

/// Where the report is written, in the app's data folder.
pub fn report_path(data_dir: &Path) -> PathBuf {
    data_dir.join("mac-bench-report.md")
}

/// `kokon-engine-mac bench` on the active profile's files (the ones `build_mac_engine_args` serves).
pub fn bench_args(profile: &ModelProfile, out: &Path, quick: bool) -> Vec<String> {
    let p = |p: &Path| p.to_string_lossy().into_owned();
    let mut a: Vec<String> = vec![
        "bench".into(),
        "--encoder".into(),
        p(&profile.text_encoder),
        "--dit".into(),
        p(&profile.diffusion_model),
        "--vae".into(),
        p(&profile.vae),
        "--out".into(),
        p(out),
    ];
    if quick {
        a.push("--quick".into());
    }
    a
}

/// Why the bench cannot run with this configuration, if it cannot.
pub fn unavailable(cfg: &AppConfig) -> Option<AppError> {
    if !cfg.uses_mac_engine() {
        return Some(AppError::invalid(
            "Turn on Kokon's Mac engine in Settings → Performance and save first.",
        ));
    }
    cfg.missing_files_error()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Progress {
    pub k: u32,
    pub n: u32,
    pub label: String,
}

/// `PROGRESS 3/7 KOKON_NA=0: accuracy at 512²` → (3, 7, label). Anything else is a human line.
pub fn parse_progress(line: &str) -> Option<Progress> {
    let rest = line.trim_end().strip_prefix("PROGRESS ")?;
    let (frac, label) = rest.split_once(' ').unwrap_or((rest, ""));
    let (k, n) = frac.split_once('/')?;
    let (k, n) = (k.parse().ok()?, n.parse().ok()?);
    (k >= 1 && k <= n).then(|| Progress {
        k,
        n,
        label: label.trim().to_string(),
    })
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum BenchState {
    Idle,
    /// Step `k` of `n` (0 before the first progress line).
    Running {
        k: u32,
        n: u32,
        label: String,
        started_at: String,
        quick: bool,
    },
    Done {
        report: String,
        finished_at: String,
        quick: bool,
    },
    Failed {
        error: String,
        log: String,
    },
}

/// A failure's message: the engine's own `error: …` line when it printed one.
fn failure_message(log: &[String], status: Option<i32>) -> String {
    if let Some(e) = log
        .iter()
        .rev()
        .find_map(|l| l.trim().strip_prefix("error: "))
    {
        let first = e.lines().next().unwrap_or(e);
        return format!("The benchmark stopped: {first}");
    }
    match status {
        Some(c) => format!("The benchmark stopped (exit code {c})."),
        None => "The benchmark stopped: the engine was killed (out of memory?).".into(),
    }
}

/// What [`MacBench::start`] runs.
pub struct BenchCommand {
    pub exe: PathBuf,
    pub args: Vec<String>,
    pub envs: Vec<(String, String)>,
    /// The file the bench writes its report to (`--out`).
    pub report: PathBuf,
    pub quick: bool,
}

pub struct MacBench {
    state: Mutex<BenchState>,
    log: Mutex<VecDeque<String>>,
    pid: Mutex<Option<u32>>,
    cancelled: AtomicBool,
}

impl MacBench {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(BenchState::Idle),
            log: Mutex::new(VecDeque::new()),
            pid: Mutex::new(None),
            cancelled: AtomicBool::new(false),
        })
    }

    pub fn state(&self) -> BenchState {
        self.state.lock().unwrap().clone()
    }

    pub fn is_running(&self) -> bool {
        matches!(*self.state.lock().unwrap(), BenchState::Running { .. })
    }

    fn push_line(&self, line: String) {
        let mut log = self.log.lock().unwrap();
        if log.len() == LOG_LINES {
            log.pop_front();
        }
        log.push_back(line);
    }

    /// Spawns the bench and follows it; `changed` gets every new state. `on_exit` runs when it ends, before the
    /// final state is published (the caller releases the server there).
    pub fn start(
        self: &Arc<Self>,
        c: BenchCommand,
        changed: impl Fn(BenchState) + Send + Sync + 'static,
        on_exit: impl FnOnce() + Send + 'static,
    ) -> AppResult<()> {
        let BenchCommand {
            exe,
            args,
            envs,
            report,
            quick,
        } = c;
        let mut cmd = tokio::process::Command::new(&exe);
        cmd.args(&args)
            .envs(envs)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        if let Some(d) = exe.parent() {
            cmd.current_dir(d);
        }
        // Its own process group, so Cancel stops the bench and the child it is running.
        #[cfg(unix)]
        cmd.process_group(0);
        let _ = std::fs::remove_file(&report);
        let mut child = cmd.spawn().map_err(|e| {
            AppError::new(
                ErrorCode::ServerStartFailed,
                format!("Could not start the benchmark: {e}"),
            )
        })?;
        self.log.lock().unwrap().clear();
        self.cancelled.store(false, Ordering::Relaxed);
        *self.pid.lock().unwrap() = child.id();
        let started_at = chrono::Utc::now().to_rfc3339();
        let running = BenchState::Running {
            k: 0,
            n: 0,
            label: "Starting".into(),
            started_at: started_at.clone(),
            quick,
        };
        *self.state.lock().unwrap() = running.clone();
        changed(running);

        let changed = Arc::new(changed);
        let (stdout, stderr) = (child.stdout.take(), child.stderr.take());
        let me = self.clone();
        tokio::spawn(async move {
            let out = {
                let (me, changed, started_at) = (me.clone(), changed.clone(), started_at.clone());
                read_lines(stdout, move |line| {
                    if let Some(p) = parse_progress(&line) {
                        let s = BenchState::Running {
                            k: p.k,
                            n: p.n,
                            label: p.label,
                            started_at: started_at.clone(),
                            quick,
                        };
                        *me.state.lock().unwrap() = s.clone();
                        changed(s);
                    }
                    me.push_line(line);
                })
            };
            let err = {
                let me = me.clone();
                read_lines(stderr, move |line| me.push_line(line))
            };
            tokio::join!(out, err);
            let status = child.wait().await.ok();
            *me.pid.lock().unwrap() = None;
            on_exit();
            let log: Vec<String> = me.log.lock().unwrap().iter().cloned().collect();
            let fin = if me.cancelled.load(Ordering::Relaxed) {
                BenchState::Idle
            } else {
                match (
                    status.filter(|s| s.success()),
                    std::fs::read_to_string(&report),
                ) {
                    (Some(_), Ok(r)) if !r.trim().is_empty() => BenchState::Done {
                        report: r,
                        finished_at: chrono::Utc::now().to_rfc3339(),
                        quick,
                    },
                    _ => BenchState::Failed {
                        error: failure_message(&log, status.and_then(|s| s.code())),
                        log: log.join("\n"),
                    },
                }
            };
            *me.state.lock().unwrap() = fin.clone();
            changed(fin);
        });
        Ok(())
    }

    /// Stops the bench and the configuration it is running; the state goes back to idle when it has exited.
    pub fn cancel(&self) {
        let Some(pid) = *self.pid.lock().unwrap() else {
            return;
        };
        self.cancelled.store(true, Ordering::Relaxed);
        #[cfg(unix)]
        {
            // The whole group (see `start`); the children also exit on their own when the bench is gone.
            let _ = std::process::Command::new("/bin/kill")
                .args(["-KILL", "--", &format!("-{pid}")])
                .status();
        }
        #[cfg(not(unix))]
        {
            let _ = std::process::Command::new("taskkill")
                .args(["/F", "/T", "/PID", &pid.to_string()])
                .status();
        }
    }

    /// Forgets a finished report or failure (the card shows Run again).
    pub fn clear(&self) {
        let mut s = self.state.lock().unwrap();
        if !matches!(*s, BenchState::Running { .. }) {
            *s = BenchState::Idle;
        }
    }
}

async fn read_lines(pipe: Option<impl AsyncRead + Unpin>, mut each: impl FnMut(String)) {
    let Some(pipe) = pipe else { return };
    let mut lines = BufReader::new(pipe).lines();
    while let Ok(Some(line)) = lines.next_line().await {
        each(line);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn progress_lines() {
        assert_eq!(
            parse_progress("PROGRESS 1/7 Accuracy reference (512², most accurate path)"),
            Some(Progress {
                k: 1,
                n: 7,
                label: "Accuracy reference (512², most accurate path)".into()
            })
        );
        assert_eq!(
            parse_progress("PROGRESS 7/7 KOKON_NA=0: two images at 1024²\n"),
            Some(Progress {
                k: 7,
                n: 7,
                label: "KOKON_NA=0: two images at 1024²".into()
            })
        );
        assert_eq!(
            parse_progress("PROGRESS 2/3"),
            Some(Progress {
                k: 2,
                n: 3,
                label: String::new()
            })
        );
        for human in [
            "[1/3] default (paths above): accuracy at 512²",
            "PROGRESS",
            "PROGRESS 4/3 x",
            "PROGRESS 0/3 x",
            "PROGRESS a/b x",
            " PROGRESS 1/2 x",
        ] {
            assert_eq!(parse_progress(human), None, "{human}");
        }
    }

    #[test]
    fn args_use_the_active_files() {
        let profile = ModelProfile {
            diffusion_model: PathBuf::from("/m/dit.gguf"),
            vae: PathBuf::from("/m/vae.safetensors"),
            text_encoder: PathBuf::from("/m/te.gguf"),
            vision_projector: Some(PathBuf::from("/m/mmproj.gguf")),
            ..ModelProfile::default()
        };
        let out = report_path(Path::new("/data"));
        assert_eq!(out, Path::new("/data").join("mac-bench-report.md"));
        let a = bench_args(&profile, &out, false);
        assert_eq!(
            a,
            [
                "bench",
                "--encoder",
                "/m/te.gguf",
                "--dit",
                "/m/dit.gguf",
                "--vae",
                "/m/vae.safetensors",
                "--out",
                out.to_str().unwrap()
            ]
        );
        assert_eq!(
            bench_args(&profile, &out, true).last().map(String::as_str),
            Some("--quick")
        );
    }

    #[test]
    fn needs_the_mac_engine() {
        let mut cfg = AppConfig::default();
        cfg.mac_engine = false;
        assert!(unavailable(&cfg).is_some_and(|e| e.message.contains("Mac engine")));
    }

    #[test]
    fn failure_messages() {
        let log = vec![
            "accuracy at 512²…".to_string(),
            "error: kernel(\"bench-run accuracy failed\")\nmore".to_string(),
        ];
        assert_eq!(
            failure_message(&log, Some(1)),
            "The benchmark stopped: kernel(\"bench-run accuracy failed\")"
        );
        assert_eq!(
            failure_message(&[], Some(3)),
            "The benchmark stopped (exit code 3)."
        );
        assert!(failure_message(&[], None).contains("killed"));
    }

    fn command(exe: &Path, args: Vec<String>, report: &Path, quick: bool) -> BenchCommand {
        BenchCommand {
            exe: exe.to_path_buf(),
            args,
            envs: vec![],
            report: report.to_path_buf(),
            quick,
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn runs_reports_and_cancels() {
        let dir = tempfile::tempdir().unwrap();
        let report = report_path(dir.path());
        let script = dir.path().join("fake-bench.sh");
        std::fs::write(&script, "#!/bin/sh\necho 'PROGRESS 1/2 one'\necho 'PROGRESS 2/2 two'\nprintf '### report' > \"$2\"\n").unwrap();
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();

        let bench = MacBench::new();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let (tx, rx) = tokio::sync::oneshot::channel();
        let s = seen.clone();
        let tx = Mutex::new(Some(tx));
        bench
            .start(
                command(
                    &script,
                    vec!["--out".into(), report.to_string_lossy().into()],
                    &report,
                    true,
                ),
                move |st| {
                    let end = !matches!(st, BenchState::Running { .. });
                    s.lock().unwrap().push(st);
                    if end && let Some(tx) = tx.lock().unwrap().take() {
                        let _ = tx.send(());
                    }
                },
                || {},
            )
            .unwrap();
        rx.await.unwrap();
        let seen = seen.lock().unwrap();
        assert!(
            seen.iter()
                .any(|s| matches!(s, BenchState::Running { k: 2, n: 2, .. }))
        );
        assert!(
            matches!(seen.last(), Some(BenchState::Done { report, quick: true, .. }) if report == "### report")
        );

        // A bench that hangs: Cancel brings it back to idle, with no failure.
        std::fs::write(&script, "#!/bin/sh\necho 'PROGRESS 1/3 one'\nsleep 30\n").unwrap();
        let (tx, rx) = tokio::sync::oneshot::channel();
        let tx = Mutex::new(Some(tx));
        bench
            .start(
                command(&script, vec![], &report, false),
                move |st| {
                    if !matches!(st, BenchState::Running { .. })
                        && let Some(tx) = tx.lock().unwrap().take()
                    {
                        let _ = tx.send(st);
                    }
                },
                || {},
            )
            .unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        bench.cancel();
        let end = tokio::time::timeout(std::time::Duration::from_secs(5), rx)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(end, BenchState::Idle);

        // One that fails: its error line and the log.
        std::fs::write(
            &script,
            "#!/bin/sh\necho 'starting'\necho 'error: no Metal device' >&2\nexit 1\n",
        )
        .unwrap();
        let (tx, rx) = tokio::sync::oneshot::channel();
        let tx = Mutex::new(Some(tx));
        bench
            .start(
                command(&script, vec![], &report, false),
                move |st| {
                    if !matches!(st, BenchState::Running { .. })
                        && let Some(tx) = tx.lock().unwrap().take()
                    {
                        let _ = tx.send(st);
                    }
                },
                || {},
            )
            .unwrap();
        match rx.await.unwrap() {
            BenchState::Failed { error, log } => {
                assert_eq!(error, "The benchmark stopped: no Metal device");
                assert!(log.contains("starting"));
            }
            other => panic!("{other:?}"),
        }
    }
}
