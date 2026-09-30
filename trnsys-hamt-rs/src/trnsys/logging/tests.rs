//! Unit tests of the logging machinery.
//!
//! They install the subscriber of [`build_subscriber`] with `tracing::subscriber::with_default`,
//! so no global state (global subscriber, environment, working directory) is involved, and they
//! run without the TRNSYS kernel: the current unit and the simulation time are injected, the log
//! files go to a private temporary directory, stdout is off and nothing is forwarded to TRNSYS
//! unless a test injects its own receiver.

use super::*;
use std::path::Path;
use std::sync::atomic::{AtomicI32, AtomicU64, AtomicUsize, Ordering};
use tracing::{debug, error, info, warn};

/// A scratch directory that is removed on drop.
struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let unique = format!(
            "trnsys_hamt_rs_logtest_{tag}_{}_{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        );
        let path = std::env::temp_dir().join(unique);
        std::fs::create_dir_all(&path).expect("cannot create the temporary log directory");
        TempDir(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }

    fn exists(&self, name: &str) -> bool {
        self.0.join(name).exists()
    }

    fn read(&self, name: &str) -> String {
        std::fs::read_to_string(self.0.join(name))
            .unwrap_or_else(|err| panic!("cannot read {name}: {err}"))
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The injected TRNSYS state (current unit, simulation time) and a private log directory.
struct Harness {
    dir: TempDir,
    unit: Arc<AtomicI32>,
    sim_time: Arc<AtomicU64>,
}

impl Harness {
    /// Starts at unit 200, `T = 12.5`.
    fn new(tag: &str) -> Self {
        Harness {
            dir: TempDir::new(tag),
            unit: Arc::new(AtomicI32::new(200)),
            sim_time: Arc::new(AtomicU64::new(12.5f64.to_bits())),
        }
    }

    fn set_unit(&self, unit: i32) {
        self.unit.store(unit, Ordering::SeqCst);
    }

    /// Deterministic timestamps, no stdout, no TRNSYS, `TRNSYS_HAMT_RS` unset (INFO).
    fn config(&self) -> LogConfig {
        let unit = self.unit.clone();
        let sim_time = self.sim_time.clone();
        LogConfig {
            dir: self.dir.path().to_path_buf(),
            unit: Arc::new(move || unit.load(Ordering::SeqCst)),
            sim_time: Arc::new(move || f64::from_bits(sim_time.load(Ordering::SeqCst))),
            timer: LogTimer::new(|w| w.write_str("TS")),
            filter: FilterSource::Directives(String::new()),
            stdout: false,
            trnsys: None,
        }
    }

    fn config_with_filter(&self, directives: &str) -> LogConfig {
        LogConfig {
            filter: FilterSource::Directives(directives.to_string()),
            ..self.config()
        }
    }
}

/// Runs `run` with the subscriber built from `config` as the thread's default.
fn with_log<R>(config: LogConfig, run: impl FnOnce() -> R) -> R {
    tracing::subscriber::with_default(build_subscriber(config), run)
}

/// What a recording TRNSYS receiver was sent: `(level, message)`.
type Recorded = Arc<Mutex<Vec<(Level, String)>>>;

/// A TRNSYS receiver that records everything it is sent.
fn recording_sink() -> (TrnsysSink, Recorded) {
    let calls: Recorded = Arc::default();
    let recorded = calls.clone();
    let sink: TrnsysSink = Arc::new(move |level, message: &str| {
        recorded.lock().unwrap().push((level, message.to_string()));
    });
    (sink, calls)
}

// ---------------------------------------------------------------------------------------------
// Bug 1: one log file per unit
// ---------------------------------------------------------------------------------------------

#[test]
fn events_of_different_units_go_to_separate_files() {
    let h = Harness::new("units");

    with_log(h.config(), || {
        h.set_unit(200);
        info!("info of wall 200");
        debug!("debug of wall 200");

        h.set_unit(201);
        info!("info of wall 201");
        debug!("debug of wall 201");

        h.set_unit(200);
        warn!("warn of wall 200");
    });

    let log200 = h.dir.read("type_395_200.log");
    let log201 = h.dir.read("type_395_201.log");

    // INFO and WARN are there, DEBUG is not (default level: INFO).
    assert!(log200.contains("info of wall 200"), "{log200}");
    assert!(log200.contains("warn of wall 200"), "{log200}");
    assert!(log201.contains("info of wall 201"), "{log201}");
    assert!(!log200.contains("debug of wall"), "{log200}");
    assert!(!log201.contains("debug of wall"), "{log201}");

    // Nothing of the other unit ...
    assert!(!log200.contains("wall 201"), "{log200}");
    assert!(!log201.contains("wall 200"), "{log201}");

    // ... and every line carries the prefix of its own unit.
    assert_eq!(log200.lines().count(), 2, "{log200}");
    assert!(
        log200
            .lines()
            .all(|l| l.starts_with("[Unit 200] [T=12.5] ")),
        "{log200}"
    );
    assert_eq!(log201.lines().count(), 1, "{log201}");
    assert!(
        log201
            .lines()
            .all(|l| l.starts_with("[Unit 201] [T=12.5] ")),
        "{log201}"
    );
}

#[test]
fn non_positive_unit_is_filed_under_unit_zero() {
    let h = Harness::new("unit_zero");

    with_log(h.config(), || {
        h.set_unit(0);
        info!("no unit is current");
        h.set_unit(-1);
        info!("negative unit");
    });

    let log = h.dir.read("type_395_0.log");
    assert!(log.contains("no unit is current"), "{log}");
    assert!(log.contains("negative unit"), "{log}");
    // The prefix still reports the raw unit.
    assert!(log.contains("[Unit -1] "), "{log}");
    assert!(!h.dir.exists("type_395_-1.log"));
    assert_eq!(unit_key(-7), 0);
    assert_eq!(log_file_name(0), "type_395_0.log");
    assert_eq!(log_file_name(203), "type_395_203.log");
}

#[test]
fn log_file_is_opened_lazily_and_truncated_on_first_use_only() {
    let h = Harness::new("lazy");
    std::fs::write(h.dir.path().join("type_395_200.log"), "STALE LINE\n").unwrap();

    with_log(h.config(), || {
        // Nothing has been logged yet: no file is touched and none is created.
        assert_eq!(h.dir.read("type_395_200.log"), "STALE LINE\n");
        assert!(!h.dir.exists("type_395_201.log"));

        info!("first");
        let after_first = h.dir.read("type_395_200.log");
        assert!(
            !after_first.contains("STALE"),
            "not truncated on first use: {after_first}"
        );
        assert!(after_first.contains("first"), "{after_first}");

        // Later events are appended, the file is not truncated again ...
        info!("second");
        let after_second = h.dir.read("type_395_200.log");
        assert!(
            after_second.contains("first") && after_second.contains("second"),
            "{after_second}"
        );
        // ... and another unit's file is still not there.
        assert!(!h.dir.exists("type_395_201.log"));

        h.set_unit(201);
        info!("other unit");
        assert!(h.dir.exists("type_395_201.log"));
    });
}

// ---------------------------------------------------------------------------------------------
// Bug 2: timer / file / line / ANSI have to reach the formatter that is really used
// ---------------------------------------------------------------------------------------------

#[test]
fn line_format_uses_prefix_timer_file_line_and_no_ansi() {
    let h = Harness::new("format");

    with_log(h.config(), || info!("value {}", 42));

    let log = h.dir.read("type_395_200.log");
    let line = log.lines().next().expect("one log line");

    // `[Unit N] [T=t]` prefix, then the configured timer (not the default UTC `SystemTime`).
    assert!(line.starts_with("[Unit 200] [T=12.5] TS "), "{line}");
    assert!(line.contains("INFO"), "{line}");
    assert!(line.contains(module_path!()), "target missing: {line}");

    // file:line
    let after_file = line
        .split("tests.rs:")
        .nth(1)
        .unwrap_or_else(|| panic!("file name missing: {line}"));
    let line_number: String = after_file
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    assert!(!line_number.is_empty(), "line number missing: {line}");
    assert!(
        after_file[line_number.len()..].starts_with(": value 42"),
        "{line}"
    );

    // no ANSI escapes
    assert!(
        !log.contains('\u{1b}'),
        "ANSI escape in the log file: {log:?}"
    );
}

#[test]
fn local_timer_writes_rfc3339() {
    let now = LogTimer::local().now();
    let bytes = now.as_bytes();

    // 2026-09-30T14:03:11.123456+02:00  (or ...Z)
    assert!(now.len() >= 20, "{now}");
    for (i, expected) in [(4, b'-'), (7, b'-'), (10, b'T'), (13, b':'), (16, b':')] {
        assert_eq!(bytes[i], expected, "{now}");
    }
    let offset_ok = now.ends_with('Z') || {
        let tail = &now.as_bytes()[now.len() - 6..];
        (tail[0] == b'+' || tail[0] == b'-') && tail[3] == b':'
    };
    assert!(offset_ok, "{now}");
}

// ---------------------------------------------------------------------------------------------
// Bug 3: the environment variable can raise (and lower) the level
// ---------------------------------------------------------------------------------------------

/// The log file of unit 200 after one DEBUG, one INFO and one WARN event with `directives`.
fn log_with_filter(tag: &str, directives: &str) -> String {
    let h = Harness::new(tag);
    with_log(h.config_with_filter(directives), || {
        debug!("a debug event");
        info!("an info event");
        warn!("a warn event");
    });
    if h.dir.exists("type_395_200.log") {
        h.dir.read("type_395_200.log")
    } else {
        String::new()
    }
}

#[test]
fn default_level_is_info() {
    // `TRNSYS_HAMT_RS` unset or empty
    let log = log_with_filter("default", "");
    assert!(!log.contains("a debug event"), "{log}");
    assert!(log.contains("an info event"), "{log}");
    assert!(log.contains("a warn event"), "{log}");
}

#[test]
fn env_directive_debug_raises_the_level() {
    let log = log_with_filter("debug", "debug");
    assert!(log.contains("a debug event"), "{log}");
    assert!(log.contains("an info event"), "{log}");
    assert!(log.contains("a warn event"), "{log}");
}

#[test]
fn env_directives_can_lower_or_target_the_level() {
    // lower
    let log = log_with_filter("warn", "warn");
    assert!(!log.contains("an info event"), "{log}");
    assert!(log.contains("a warn event"), "{log}");

    // per crate, the form documented in CLAUDE.md (target = module path of this crate)
    let log = log_with_filter("crate_debug", "trnsys_hamt_rs=debug,hamt_core=debug");
    assert!(log.contains("a debug event"), "{log}");

    // a directive for another target only: nothing of this crate is enabled any more, exactly
    // like `RUST_LOG=other=debug`. (Combine with a base level: `info,hamt_core=debug`.)
    let log = log_with_filter("other_target", "hamt_core=debug");
    assert!(log.is_empty(), "{log}");
    let log = log_with_filter("base_plus_target", "info,hamt_core=debug");
    assert!(
        log.contains("an info event") && !log.contains("a debug event"),
        "{log}"
    );

    // garbage falls back to the default
    let log = log_with_filter("garbage", "this is=not a level");
    assert!(
        log.contains("an info event") && !log.contains("a debug event"),
        "{log}"
    );
}

// ---------------------------------------------------------------------------------------------
// Forwarding to TRNSYS
// ---------------------------------------------------------------------------------------------

#[test]
fn trnsys_forwarding_honours_env_filter_and_threshold() {
    let h = Harness::new("forward");

    // The env filter lets everything through, the threshold (INFO) cuts DEBUG.
    let (sink, calls) = recording_sink();
    let config = LogConfig {
        trnsys: Some(TrnsysForward {
            threshold: Level::INFO,
            sink,
        }),
        ..h.config_with_filter("debug")
    };
    with_log(config, || {
        debug!("d");
        info!("i");
        warn!("w");
        error!("e");
    });
    {
        let calls = calls.lock().unwrap();
        let levels: Vec<Level> = calls.iter().map(|(level, _)| *level).collect();
        assert_eq!(
            levels,
            [Level::INFO, Level::WARN, Level::ERROR],
            "{calls:?}"
        );
        assert!(calls[0].1.starts_with("INFO "), "{:?}", calls[0]);
        assert!(calls[0].1.ends_with(": i"), "{:?}", calls[0]);
    }

    // The env filter can narrow what TRNSYS hears (`warn`), although the threshold would allow more.
    let (sink, calls) = recording_sink();
    let config = LogConfig {
        trnsys: Some(TrnsysForward {
            threshold: Level::DEBUG,
            sink,
        }),
        ..h.config_with_filter("warn")
    };
    with_log(config, || {
        debug!("d");
        info!("i");
        warn!("w");
        error!("e");
    });
    let levels: Vec<Level> = calls
        .lock()
        .unwrap()
        .iter()
        .map(|(level, _)| *level)
        .collect();
    assert_eq!(levels, [Level::WARN, Level::ERROR]);

    // Default filter (INFO) with a DEBUG threshold: the filter still wins.
    let (sink, calls) = recording_sink();
    let config = LogConfig {
        trnsys: Some(TrnsysForward {
            threshold: Level::DEBUG,
            sink,
        }),
        ..h.config()
    };
    with_log(config, || {
        debug!("d");
        info!("i");
    });
    let levels: Vec<Level> = calls
        .lock()
        .unwrap()
        .iter()
        .map(|(level, _)| *level)
        .collect();
    assert_eq!(levels, [Level::INFO]);
}
