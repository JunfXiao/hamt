//! Unit tests of the logging machinery.
//!
//! They install the subscriber of [`build_subscriber`] with `tracing::subscriber::with_default`,
//! so no global state (global subscriber, environment, working directory) is involved, and they
//! run without the TRNSYS kernel: the current unit and the simulation time are injected, the log
//! files go to a private temporary directory, stdout is off and nothing is forwarded to TRNSYS
//! unless a test injects its own receiver.

use super::ring::{DEFAULT_RING_MB, RingSetting};
use super::*;
use std::path::Path;
use std::sync::PoisonError;
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

    /// Deterministic timestamps, no stdout, no TRNSYS, no debug ring, `TRNSYS_HAMT_RS` unset (INFO).
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
            ring_bytes: None,
        }
    }

    fn config_with_filter(&self, directives: &str) -> LogConfig {
        LogConfig {
            filter: FilterSource::Directives(directives.to_string()),
            ..self.config()
        }
    }

    /// [`config`](Self::config) with a debug ring of `ring_bytes` per unit.
    fn config_with_ring(&self, ring_bytes: usize) -> LogConfig {
        LogConfig {
            ring_bytes: Some(ring_bytes),
            ..self.config()
        }
    }
}

/// Runs `run` with the subscriber built from `config` as the thread's default.
fn with_log<R>(config: LogConfig, run: impl FnOnce() -> R) -> R {
    let (subscriber, _handle) = build_subscriber(config);
    tracing::subscriber::with_default(subscriber, run)
}

/// Checks that `timestamp` looks like `2026-09-30T14:03:11.123456+02:00` (or `...Z`).
fn assert_rfc3339(timestamp: &str) {
    let bytes = timestamp.as_bytes();

    assert!(timestamp.len() >= 20, "{timestamp}");
    for (i, expected) in [(4, b'-'), (7, b'-'), (10, b'T'), (13, b':'), (16, b':')] {
        assert_eq!(bytes[i], expected, "{timestamp}");
    }
    let offset_ok = timestamp.ends_with('Z') || {
        let tail = &bytes[bytes.len() - 6..];
        (tail[0] == b'+' || tail[0] == b'-') && tail[3] == b':'
    };
    assert!(offset_ok, "{timestamp}");
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
    assert_rfc3339(&LogTimer::local().now());
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

// ---------------------------------------------------------------------------------------------
// Debug ring buffer (flight recorder)
// ---------------------------------------------------------------------------------------------

/// A parsed `type_395_<unit>_debug.log`.
struct Dump {
    reason: String,
    written: String,
    unit: i32,
    entries: usize,
    bytes: usize,
    discarded: u64,
    lines: Vec<String>,
}

/// Parses a dump file and checks its four header lines on the way.
fn parse_dump(text: &str) -> Dump {
    let mut lines = text.lines();
    let mut header = |prefix: &str| -> String {
        let line = lines
            .next()
            .unwrap_or_else(|| panic!("header line `{prefix}` missing"));
        line.strip_prefix(prefix)
            .unwrap_or_else(|| panic!("expected `{prefix}...`, got `{line}`"))
            .to_string()
    };

    header("# trnsys-hamt-rs debug ring buffer dump");
    let reason = header("# reason: ");
    let written = header("# written: ");
    // 201, entries: 3 (1234 bytes), older entries discarded: 0
    let counts = header("# unit: ");
    let (unit, rest) = counts.split_once(", entries: ").expect("entries");
    let (entries, rest) = rest.split_once(" (").expect("entries (bytes)");
    let (bytes, discarded) = rest
        .split_once(" bytes), older entries discarded: ")
        .expect("discarded");

    Dump {
        reason,
        written,
        unit: unit.parse().expect("unit"),
        entries: entries.parse().expect("entries"),
        bytes: bytes.parse().expect("bytes"),
        discarded: discarded.parse().expect("discarded"),
        lines: lines.map(str::to_string).collect(),
    }
}

/// Names of the files in the log directory that contain `part`.
fn files_containing(dir: &TempDir, part: &str) -> Vec<String> {
    std::fs::read_dir(dir.path())
        .expect("read the log directory")
        .filter_map(|entry| entry.ok()?.file_name().into_string().ok())
        .filter(|name| name.contains(part))
        .collect()
}

#[test]
fn ring_keeps_debug_lines_and_evicts_the_oldest_past_the_budget() {
    let h = Harness::new("ring_evict");
    let budget = 3_000;
    let (subscriber, handle) = build_subscriber(h.config_with_ring(budget));

    tracing::subscriber::with_default(subscriber, || {
        for i in 0..100 {
            debug!("event {i:03}");
        }
        info!("final info");
    });

    // DEBUG lives in the ring only: the log file (INFO and above) has just the INFO line.
    let log = h.dir.read("type_395_200.log");
    assert_eq!(log.lines().count(), 1, "{log}");
    assert!(log.contains("final info"), "{log}");

    handle.dump_debug_rings("manual dump");
    let dump = parse_dump(&h.dir.read("type_395_200_debug.log"));
    assert_eq!(dump.reason, "manual dump");
    assert_eq!(dump.unit, 200);

    // 100 DEBUG lines and 1 INFO line were recorded; the oldest were evicted to fit the budget,
    // and their number is counted.
    assert_eq!(
        dump.entries as u64 + dump.discarded,
        101,
        "entries + discarded"
    );
    assert!(dump.discarded > 0, "the budget must have forced evictions");
    assert!(dump.entries > 1);
    assert!(
        dump.bytes <= budget,
        "{} bytes > budget {budget}",
        dump.bytes
    );
    assert_eq!(dump.lines.len(), dump.entries);
    assert_eq!(
        dump.lines.iter().map(|line| line.len() + 1).sum::<usize>(),
        dump.bytes
    );

    // What is left is the newest history, in order and without gaps: DEBUG lines ...
    let (last, debug_lines) = dump.lines.split_last().unwrap();
    for (k, line) in debug_lines.iter().enumerate() {
        let expected = format!("event {:03}", dump.discarded as usize + k);
        assert!(
            line.contains(" DEBUG ") && line.ends_with(&expected),
            "{line} vs {expected}"
        );
    }
    // ... up to and including the INFO line.
    assert!(
        last.contains(" INFO ") && last.ends_with("final info"),
        "{last}"
    );
}

#[test]
fn error_event_dumps_the_ring_of_every_unit() {
    let h = Harness::new("ring_error");
    let config = LogConfig {
        timer: LogTimer::local(),
        ..h.config_with_ring(1 << 20)
    };
    let (subscriber, _handle) = build_subscriber(config);

    tracing::subscriber::with_default(subscriber, || {
        h.set_unit(200);
        debug!("debug of wall 200");
        info!("info of wall 200");

        h.set_unit(201);
        debug!("first debug of wall 201");
        debug!("second debug of wall 201");
        assert!(
            files_containing(&h.dir, "_debug").is_empty(),
            "nothing is dumped before an error"
        );

        error!("boom {}", 42);

        // The dump is on disk when `error!` returns.
        assert!(h.dir.exists("type_395_201_debug.log"));
    });

    // Unit 201: header, the DEBUG lines that led up to the error, and the error line itself.
    let dump201 = parse_dump(&h.dir.read("type_395_201_debug.log"));
    assert!(
        dump201.reason.starts_with("ERROR event"),
        "{}",
        dump201.reason
    );
    assert!(dump201.reason.contains("boom 42"), "{}", dump201.reason);
    assert_rfc3339(&dump201.written);
    assert_eq!(
        (dump201.unit, dump201.entries, dump201.discarded),
        (201, 3, 0)
    );
    assert_eq!(dump201.lines.len(), 3, "{:?}", dump201.lines);
    assert!(
        dump201.lines[0].contains(" DEBUG ")
            && dump201.lines[0].ends_with("first debug of wall 201")
    );
    assert!(
        dump201.lines[1].contains(" DEBUG ")
            && dump201.lines[1].ends_with("second debug of wall 201")
    );
    assert!(dump201.lines[2].contains(" ERROR ") && dump201.lines[2].ends_with("boom 42"));
    assert!(
        dump201
            .lines
            .iter()
            .all(|line| line.starts_with("[Unit 201] "))
    );

    // Unit 200's dump is written too.
    let dump200 = parse_dump(&h.dir.read("type_395_200_debug.log"));
    assert_eq!(
        (dump200.unit, dump200.entries, dump200.discarded),
        (200, 2, 0)
    );
    assert!(
        dump200.lines[0].ends_with("debug of wall 200"),
        "{:?}",
        dump200.lines
    );
    assert!(
        dump200.lines[1].ends_with("info of wall 200"),
        "{:?}",
        dump200.lines
    );

    // The error is in the log file of unit 201 as well, the DEBUG lines are not.
    let log201 = h.dir.read("type_395_201.log");
    assert!(
        log201.contains("boom 42") && !log201.contains("debug of wall"),
        "{log201}"
    );
}

#[test]
fn error_dump_is_written_before_trnsys_is_notified() {
    let h = Harness::new("ring_order");
    let dump_path = h.dir.path().join("type_395_200_debug.log");
    let log_path = h.dir.path().join("type_395_200.log");

    // What is on disk at the moment TRNSYS is told about the error.
    let on_disk_then: Arc<Mutex<Option<(String, String)>>> = Arc::default();
    let sink: TrnsysSink = {
        let on_disk_then = on_disk_then.clone();
        Arc::new(move |level, _message: &str| {
            if level == Level::ERROR {
                let dump = std::fs::read_to_string(&dump_path).unwrap_or_default();
                let log = std::fs::read_to_string(&log_path).unwrap_or_default();
                *on_disk_then.lock().unwrap() = Some((dump, log));
            }
        })
    };
    let config = LogConfig {
        trnsys: Some(TrnsysForward {
            threshold: Level::INFO,
            sink,
        }),
        ..h.config_with_ring(1 << 20)
    };

    with_log(config, || {
        debug!("history before the error");
        error!("fatal error");
    });

    let (dump, log) = on_disk_then
        .lock()
        .unwrap()
        .take()
        .expect("TRNSYS was told about the error");
    assert!(
        dump.contains("history before the error"),
        "no dump yet when TRNSYS was told: {dump:?}"
    );
    assert!(
        dump.contains("fatal error"),
        "the dump lacks the error line: {dump:?}"
    );
    assert!(
        log.contains("fatal error"),
        "the log file lacks the error line: {log:?}"
    );
}

#[test]
fn ring_env_setting_parsing() {
    let default = RingSetting {
        budget_bytes: Some(DEFAULT_RING_MB * 1024 * 1024),
        warning: None,
    };
    let mib = |n: usize| RingSetting {
        budget_bytes: Some(n * 1024 * 1024),
        warning: None,
    };
    let disabled = RingSetting {
        budget_bytes: None,
        warning: None,
    };

    // not set, or empty: 8 MiB, silently
    assert_eq!(DEFAULT_RING_MB, 8);
    assert_eq!(parse_ring_setting(None), default);
    assert_eq!(parse_ring_setting(Some("")), default);
    assert_eq!(parse_ring_setting(Some("   ")), default);

    // an integer number of MiB
    assert_eq!(parse_ring_setting(Some("1")), mib(1));
    assert_eq!(parse_ring_setting(Some("32")), mib(32));
    assert_eq!(parse_ring_setting(Some(" 16 ")), mib(16));
    assert_eq!(parse_ring_setting(Some("008")), mib(8));

    // 0 switches the ring off
    assert_eq!(parse_ring_setting(Some("0")), disabled);
    assert_eq!(parse_ring_setting(Some(" 0 ")), disabled);

    // invalid: the default, with a warning that names the variable and the value
    for invalid in [
        "abc",
        "-1",
        "1.5",
        "8MB",
        "0x10",
        "1e3",
        "18446744073709551616",          // does not fit an integer
        "17592186044416",                // 2^44 MiB: the byte count overflows
        "99999999999999999999999999999", // absurd
    ] {
        let setting = parse_ring_setting(Some(invalid));
        assert_eq!(setting.budget_bytes, default.budget_bytes, "{invalid}");
        let warning = setting
            .warning
            .unwrap_or_else(|| panic!("no warning for {invalid:?}"));
        assert!(
            warning.contains(ENV_RING_MB) && warning.contains(invalid),
            "{warning}"
        );
    }
}

#[test]
fn disabled_ring_installs_no_layer_and_keeps_debug_off() {
    let h = Harness::new("ring_off");
    let (subscriber, handle) = build_subscriber(h.config());

    tracing::subscriber::with_default(subscriber, || {
        // DEBUG callsites are disabled (zero cost) ...
        assert!(
            !tracing::event_enabled!(Level::DEBUG),
            "DEBUG must stay off without a ring"
        );
        assert!(tracing::event_enabled!(Level::INFO));
        debug!("nobody records this");
        error!("boom");
    });

    // ... and there is nothing to dump, even after an error.
    handle.dump_debug_rings("manual");
    handle.dump_debug_rings_once("manual");
    assert!(files_containing(&h.dir, "_debug").is_empty());
    assert!(h.dir.read("type_395_200.log").contains("boom"));

    // With a ring, DEBUG callsites are on.
    let (subscriber, _handle) = build_subscriber(h.config_with_ring(1 << 20));
    tracing::subscriber::with_default(subscriber, || {
        assert!(
            tracing::event_enabled!(Level::DEBUG),
            "the ring needs DEBUG events"
        );
    });
}

#[test]
fn ring_does_not_change_what_the_log_file_and_trnsys_receive() {
    let h = Harness::new("ring_isolation");
    let (sink, calls) = recording_sink();
    let config = LogConfig {
        trnsys: Some(TrnsysForward {
            threshold: Level::DEBUG,
            sink,
        }),
        ..h.config_with_ring(1 << 20)
    };
    let (subscriber, handle) = build_subscriber(config);

    tracing::subscriber::with_default(subscriber, || {
        debug!("debug only for the ring");
        info!("info for everybody");
    });

    // The ring made DEBUG callsites live, but the file and TRNSYS keep their own (default) filter.
    let log = h.dir.read("type_395_200.log");
    assert!(!log.contains("debug only for the ring"), "{log}");
    assert!(log.contains("info for everybody"), "{log}");
    let levels: Vec<Level> = calls
        .lock()
        .unwrap()
        .iter()
        .map(|(level, _)| *level)
        .collect();
    assert_eq!(levels, [Level::INFO]);

    // The ring has both.
    handle.dump_debug_rings("check");
    let dump = parse_dump(&h.dir.read("type_395_200_debug.log"));
    assert_eq!(dump.entries, 2, "{:?}", dump.lines);
    assert!(dump.lines[0].ends_with("debug only for the ring"));
}

#[test]
fn ring_dump_reason_is_reduced_to_its_first_line() {
    let h = Harness::new("ring_reason");
    let (subscriber, handle) = build_subscriber(h.config_with_ring(1 << 20));

    tracing::subscriber::with_default(subscriber, || {
        debug!("something to dump");

        handle.dump_debug_rings("first line\nsecond line\r\nthird");
        assert_eq!(
            parse_dump(&h.dir.read("type_395_200_debug.log")).reason,
            "first line"
        );

        let long = "x".repeat(1_000);
        handle.dump_debug_rings(&long);
        let reason = parse_dump(&h.dir.read("type_395_200_debug.log")).reason;
        assert_eq!(
            reason.chars().count(),
            303,
            "300 characters and `...`: {reason}"
        );
        assert!(reason.starts_with(&"x".repeat(300)) && reason.ends_with("..."));

        handle.dump_debug_rings("");
        assert_eq!(parse_dump(&h.dir.read("type_395_200_debug.log")).reason, "");
    });
}

#[test]
fn ring_dump_of_a_non_positive_unit_goes_to_unit_zero() {
    let h = Harness::new("ring_unit_zero");
    let (subscriber, handle) = build_subscriber(h.config_with_ring(1 << 20));

    tracing::subscriber::with_default(subscriber, || {
        h.set_unit(0);
        debug!("outside any unit");
        h.set_unit(-3);
        debug!("also outside any unit");
    });

    handle.dump_debug_rings("manual");
    let dump = parse_dump(&h.dir.read("type_395_0_debug.log"));
    assert_eq!((dump.unit, dump.entries), (0, 2));
    assert_eq!(files_containing(&h.dir, "_debug"), ["type_395_0_debug.log"]);
    assert_eq!(debug_dump_file_name(200), "type_395_200_debug.log");
    assert_eq!(debug_dump_file_name(-1), "type_395_0_debug.log");
}

#[test]
fn dump_once_writes_only_the_first_time() {
    let h = Harness::new("ring_once");
    let (subscriber, handle) = build_subscriber(h.config_with_ring(1 << 20));
    let dump_path = h.dir.path().join("type_395_200_debug.log");

    tracing::subscriber::with_default(subscriber, || debug!("history"));

    handle.dump_debug_rings_once("first");
    assert_eq!(
        parse_dump(&h.dir.read("type_395_200_debug.log")).reason,
        "first"
    );

    std::fs::remove_file(&dump_path).unwrap();
    handle.dump_debug_rings_once("second");
    assert!(!dump_path.exists(), "the second once-dump must do nothing");

    // an ordinary dump is not affected by the guard
    handle.dump_debug_rings("third");
    assert_eq!(
        parse_dump(&h.dir.read("type_395_200_debug.log")).reason,
        "third"
    );
}

#[test]
fn last_call_dump_does_not_replace_the_dump_of_an_error() {
    let h = Harness::new("ring_once_after_error");
    let (subscriber, handle) = build_subscriber(h.config_with_ring(1 << 20));

    tracing::subscriber::with_default(subscriber, || {
        debug!("history");
        error!("the actual failure");
        debug!("logged after the failure");
    });
    let after_error = h.dir.read("type_395_200_debug.log");

    // The last call of the simulation (TRNSYS reports the error) has nothing to add: the dump
    // taken when the failure happened names its cause and must stay.
    handle.dump_debug_rings_once("last call of the simulation, TRNSYS reports an error");

    assert_eq!(h.dir.read("type_395_200_debug.log"), after_error);
    let dump = parse_dump(&after_error);
    assert!(
        dump.reason.contains("the actual failure"),
        "{}",
        dump.reason
    );
    assert!(!after_error.contains("logged after the failure"));
}

#[test]
fn lock_bounded_recovers_poison_and_gives_up_on_a_held_lock() {
    let mutex = Mutex::new(5);

    // poison it
    let poisoner = std::thread::scope(|scope| {
        scope
            .spawn(|| {
                let _guard = mutex.lock().unwrap();
                panic!("poison the mutex");
            })
            .join()
    });
    assert!(poisoner.is_err() && mutex.is_poisoned());

    // A poisoned mutex is recovered, not unwrapped.
    {
        let guard = lock_bounded(&mutex).expect("a poisoned mutex must be recovered");
        assert_eq!(*guard, 5);
    }

    // A mutex that this very thread already holds (the panic-hook situation) would deadlock
    // `lock()`; `lock_bounded` gives up instead, after roughly LOCK_PATIENCE.
    let held = lock_bounded(&mutex).unwrap();
    let started = Instant::now();
    assert!(lock_bounded(&mutex).is_none());
    let waited = started.elapsed();
    assert!(waited >= LOCK_PATIENCE, "gave up too early: {waited:?}");
    assert!(
        waited < Duration::from_secs(5),
        "waited too long: {waited:?}"
    );
    drop(held);

    // and it is usable again afterwards
    assert!(lock_bounded(&mutex).is_some());
}

#[test]
fn ring_survives_poisoned_and_held_locks() {
    let h = Harness::new("ring_locks");
    let (subscriber, handle) = build_subscriber(h.config_with_ring(1 << 20));
    let rings = handle.rings.clone().expect("the ring is enabled");
    let dump_path = h.dir.path().join("type_395_200_debug.log");

    tracing::subscriber::with_default(subscriber, || {
        debug!("before the panic");

        // A panic while the ring lock is held poisons it ...
        let outcome = std::thread::scope(|scope| {
            scope
                .spawn(|| {
                    let _guard = rings.rings.lock().unwrap();
                    panic!("poison the ring lock");
                })
                .join()
        });
        assert!(outcome.is_err() && rings.rings.is_poisoned());

        // ... but logging and dumping carry on.
        debug!("after the panic");
        handle.dump_debug_rings("poisoned");
        let dump = h.dir.read("type_395_200_debug.log");
        assert!(
            dump.contains("before the panic") && dump.contains("after the panic"),
            "{dump}"
        );

        // The ring lock is held by this very thread (a panic hook logging from inside logging
        // code): nothing may block. The records are dropped instead.
        std::fs::remove_file(&dump_path).unwrap();
        let held = rings.rings.lock().unwrap_or_else(PoisonError::into_inner);
        let started = Instant::now();
        handle.dump_debug_rings("must not deadlock");
        debug!("dropped instead of blocking");
        error!("an error must not block either");
        let waited = started.elapsed();
        assert!(
            waited < Duration::from_secs(10),
            "logging blocked for {waited:?}"
        );
        assert!(
            !dump_path.exists(),
            "nothing can be dumped while the ring is locked"
        );
        drop(held);

        // Once the lock is free everything works again.
        debug!("recovered");
        handle.dump_debug_rings("recovered");
        assert!(h.dir.read("type_395_200_debug.log").contains("recovered"));
    });

    // The error still reached the log file.
    assert!(
        h.dir
            .read("type_395_200.log")
            .contains("an error must not block either")
    );
}

#[test]
fn ring_counts_stay_consistent_under_concurrent_events() {
    const THREADS: usize = 8;
    const PER_THREAD: usize = 2_000;

    let h = Harness::new("ring_threads");
    let budget = 20_000;
    let (subscriber, handle) = build_subscriber(h.config_with_ring(budget));
    let dispatch = tracing::Dispatch::new(subscriber);

    std::thread::scope(|scope| {
        for thread in 0..THREADS {
            let dispatch = dispatch.clone();
            scope.spawn(move || {
                tracing::dispatcher::with_default(&dispatch, || {
                    for i in 0..PER_THREAD {
                        debug!(thread, "event {i}");
                    }
                });
            });
        }
    });

    handle.dump_debug_rings("threads");
    let dump = parse_dump(&h.dir.read("type_395_200_debug.log"));

    assert_eq!(
        dump.entries as u64 + dump.discarded,
        (THREADS * PER_THREAD) as u64
    );
    assert!(
        dump.bytes <= budget,
        "{} bytes > budget {budget}",
        dump.bytes
    );
    assert_eq!(dump.lines.len(), dump.entries);
    // Lines of different threads are never interleaved.
    for line in &dump.lines {
        assert!(line.starts_with("[Unit 200] [T=12.5] TS "), "{line}");
        assert!(
            line.contains(" event ") && line.contains(" thread="),
            "{line}"
        );
    }
}
