//! Logging for the Type 395 DLL.
//!
//! One TRNSYS deck usually contains several Type 395 units (one per wall), all served by the same
//! loaded DLL, and TRNSYS calls them one after the other on one thread. Every event is therefore
//! routed by the unit that is *current when the event is written*:
//!
//! * `type_395_<unit>.log` in the process working directory, one file per unit, opened lazily and
//!   truncated on the unit's first log line of the run (a non-positive unit goes to
//!   `type_395_0.log`). Every line starts with `[Unit N] [T=t]`. Events are also echoed to stdout.
//! * Events at or above [`TRNSYS_LOG_LEVEL`] are forwarded to the TRNSYS log via
//!   [`log_in_trnsys`]. **`ERROR` maps to a TRNSYS fatal error, i.e. every `error!` stops the
//!   simulation.**
//!
//! The level is controlled by the `TRNSYS_HAMT_RS` environment variable (`tracing` `EnvFilter`
//! directives, default `info`). Each of the two sinks above has its own filter instance.
//!
//! Everything that does not need the TRNSYS kernel lives behind [`build_subscriber`], which takes
//! the current unit and the simulation time as injected providers, so it can be unit-tested
//! without TRNSYS. [`init_tracing`] wires it to the kernel.

mod files;
mod format;
#[cfg(test)]
mod tests;

use crate::TYPE_NUMBER;
use crate::trnsys::{Severity, get_current_unit, get_simulation_time, log_message};
use files::{FileSink, UnitFiles};
use format::{LogTimer, event_format};
use std::fmt::{self, Write as _};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard, TryLockError};
use std::time::{Duration, Instant};
use tracing::field::{Field, Visit};
use tracing::{Event, Level, Subscriber, error};
use tracing_subscriber::filter::{EnvFilter, LevelFilter};
use tracing_subscriber::layer::{Context, Layer, SubscriberExt};
use tracing_subscriber::registry::Registry;

/// Environment variable holding the `EnvFilter` directives (e.g. `debug`, or
/// `info,hamt_core=debug`).
pub const ENV_FILTER: &str = "TRNSYS_HAMT_RS";

/// The threshold level for trnsys logging.
#[cfg(debug_assertions)]
const TRNSYS_LOG_LEVEL: Level = Level::DEBUG;
/// The threshold level for trnsys logging.
#[cfg(not(debug_assertions))]
const TRNSYS_LOG_LEVEL: Level = Level::INFO;

/// Provider of the current TRNSYS unit number.
pub(crate) type UnitFn = Arc<dyn Fn() -> i32 + Send + Sync>;
/// Provider of the current simulation time (hours).
pub(crate) type TimeFn = Arc<dyn Fn() -> f64 + Send + Sync>;
/// Receiver of the events forwarded to TRNSYS.
pub(crate) type TrnsysSink = Arc<dyn Fn(Level, &str) + Send + Sync>;

type BoxedLayer = Box<dyn Layer<Registry> + Send + Sync>;

/// Custom function to handle trnsys logging.
///
/// # Arguments
///
/// * `level` - The level of the event.
/// * `name` - The name of the event.
/// * `message` - The message of the event.
pub fn log_in_trnsys(level: Level, error_code: Option<i32>, message: &str) {
    // Perform necessary actions here, such as sending notifications, writing to a dedicated log, raising alarms, etc.
    let trnsys_severity = match level {
        Level::ERROR => Severity::Fatal,
        Level::WARN => Severity::Warning,
        _ => Severity::Notice,
    };
    let error_code = error_code.unwrap_or(-1);

    log_message(trnsys_severity, error_code, message);
}

/// Where the `TRNSYS_HAMT_RS` directives come from.
pub(crate) enum FilterSource {
    /// Read the `TRNSYS_HAMT_RS` environment variable (the DLL).
    Env,
    /// Use these directives instead of the environment (unit tests: no process-global state).
    Directives(String),
}

/// Events forwarded to TRNSYS.
pub(crate) struct TrnsysForward {
    /// Only events at or above this severity are forwarded (`Level::INFO`: error, warn, info).
    pub threshold: Level,
    /// Receives the forwarded events.
    pub sink: TrnsysSink,
}

/// Everything [`build_subscriber`] needs.
pub(crate) struct LogConfig {
    /// Directory of the `type_395_<unit>.log` files.
    pub dir: PathBuf,
    /// The current TRNSYS unit.
    pub unit: UnitFn,
    /// The current simulation time [h].
    pub sim_time: TimeFn,
    /// Timestamp source of the log lines.
    pub timer: LogTimer,
    /// Source of the `EnvFilter` directives.
    pub filter: FilterSource,
    /// Echo the log lines to stdout.
    pub stdout: bool,
    /// Forward events to TRNSYS (`None`: do not).
    pub trnsys: Option<TrnsysForward>,
}

impl LogConfig {
    /// The configuration of the DLL: working directory, TRNSYS kernel providers, environment
    /// variables, stdout and TRNSYS forwarding on.
    pub(crate) fn production() -> Self {
        LogConfig {
            dir: std::env::current_dir().unwrap_or_else(|_| std::env::temp_dir()),
            unit: Arc::new(get_current_unit),
            sim_time: Arc::new(get_simulation_time),
            timer: LogTimer::local(),
            filter: FilterSource::Env,
            stdout: true,
            trnsys: Some(TrnsysForward {
                threshold: TRNSYS_LOG_LEVEL,
                sink: Arc::new(|level, message| log_in_trnsys(level, None, message)),
            }),
        }
    }
}

/// The unit under which events are filed: non-positive units (no unit is current) all go to 0.
pub(crate) fn unit_key(unit: i32) -> i32 {
    unit.max(0)
}

/// `type_395_<unit>.log`
pub(crate) fn log_file_name(unit: i32) -> String {
    format!("type_{}_{}.log", TYPE_NUMBER, unit_key(unit))
}

/// Best-effort diagnostic on stderr for problems of the logging machinery itself. Never panics
/// (a closed or failing stderr is ignored), never logs through `tracing`.
pub(crate) fn diag(message: fmt::Arguments<'_>) {
    use std::io::Write;
    let _ = writeln!(std::io::stderr(), "[trnsys-hamt-rs] {message}");
}

/// How long logging code waits for a contended lock before it gives up on one record.
const LOCK_PATIENCE: Duration = Duration::from_millis(50);

/// Locks `mutex` for logging code, which must never deadlock or panic.
///
/// * A poisoned mutex is recovered (`into_inner`): a panic elsewhere must not silence logging,
///   and the panic hook itself logs.
/// * A mutex that stays locked for [`LOCK_PATIENCE`] yields `None` (the caller drops the record)
///   instead of blocking. This covers a panic that unwinds through logging code while it holds the
///   lock: the panic hook then logs from the very thread that owns the lock.
pub(crate) fn lock_bounded<T>(mutex: &Mutex<T>) -> Option<MutexGuard<'_, T>> {
    match mutex.try_lock() {
        Ok(guard) => return Some(guard),
        Err(TryLockError::Poisoned(poisoned)) => return Some(poisoned.into_inner()),
        Err(TryLockError::WouldBlock) => {}
    }

    let started = Instant::now();
    loop {
        std::thread::yield_now();
        match mutex.try_lock() {
            Ok(guard) => return Some(guard),
            Err(TryLockError::Poisoned(poisoned)) => return Some(poisoned.into_inner()),
            Err(TryLockError::WouldBlock) => {
                if started.elapsed() >= LOCK_PATIENCE {
                    return None;
                }
            }
        }
    }
}

struct MessageCollector {
    message: Option<String>,
    fields: Vec<(String, String)>,
}

impl MessageCollector {
    fn new() -> Self {
        MessageCollector {
            message: None,
            fields: Vec::new(),
        }
    }

    fn get_formatted_message(&self) -> Option<String> {
        if let Some(message) = &self.message {
            let mut formatted_message = message.clone();
            for (name, value) in &self.fields {
                formatted_message = formatted_message.replace(&format!("{{{}}}", name), value);
            }
            Some(formatted_message)
        } else {
            None
        }
    }
}

impl Visit for MessageCollector {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        let val_str = format!("{:?}", value);
        if field.name() == "message" {
            // message字段通常是一个静态字符串，如 "a: {a}"
            self.message = Some(val_str.trim_matches('"').to_string());
        } else {
            // 其他字段存储参数值
            self.fields.push((field.name().to_string(), val_str));
        }
    }
}

/// Custom Layer to intercept high-priority events.
struct TrnSysLogLayer {
    threshold: Level,
    sink: TrnsysSink,
}

impl TrnSysLogLayer {
    pub fn new(forward: TrnsysForward) -> Self {
        TrnSysLogLayer {
            threshold: forward.threshold,
            sink: forward.sink,
        }
    }
}

impl<S: Subscriber> Layer<S> for TrnSysLogLayer {
    /// Intercepts events and calls the custom function if the event level is above the threshold.
    ///
    /// # Arguments
    ///
    /// * `event` - The event to be processed.
    /// * `_ctx` - The context of the subscriber.
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let metadata = event.metadata();
        if metadata.level() <= &self.threshold {
            let mut text = String::new();

            let mut collector = MessageCollector::new();
            event.record(&mut collector);
            let formatted_msg = collector.get_formatted_message();
            if let Some(msg) = formatted_msg {
                let _ = write!(
                    text,
                    "{} {} {}: {}",
                    metadata.level(),
                    metadata.target(),
                    metadata.name(),
                    msg
                );
            }

            (self.sink)(*metadata.level(), &text);
        }
    }
}

/// A fresh `EnvFilter` for one layer (`EnvFilter` is not `Clone`, and every layer needs its own).
///
/// Default `INFO`; `TRNSYS_HAMT_RS=debug` really raises the level. (The former global
/// `EnvFilter::from_env(..).add_directive("info")` could never do that: a directive of equal
/// specificity replaces the earlier one, so the appended `info` overrode `debug`.)
fn env_filter(source: &FilterSource) -> EnvFilter {
    let builder = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .with_env_var(ENV_FILTER);

    match source {
        FilterSource::Env => builder.from_env_lossy(),
        FilterSource::Directives(directives) => builder.parse_lossy(directives),
    }
}

/// Builds the subscriber of the DLL from `config`; it touches no global state.
///
/// Every layer carries its own filter (no global filter), so a layer never sees events that were
/// not meant for it:
///
/// 1. the unit log file (and stdout), filtered by `TRNSYS_HAMT_RS`,
/// 2. the forwarding to TRNSYS, filtered by `TRNSYS_HAMT_RS` and then by its own threshold.
///
/// Inner layers receive an event first (they are earlier in the list).
pub(crate) fn build_subscriber(config: LogConfig) -> impl Subscriber + Send + Sync + 'static {
    let LogConfig {
        dir,
        unit,
        sim_time,
        timer,
        filter,
        stdout,
        trnsys,
    } = config;

    let files = Arc::new(UnitFiles::new(dir, unit.clone()));
    let mut layers: Vec<BoxedLayer> = Vec::new();

    layers.push(
        tracing_subscriber::fmt::layer::<Registry>()
            .with_writer(FileSink::new(files, stdout))
            .with_ansi(false)
            .event_format(event_format(&timer, &unit, &sim_time))
            .with_filter(env_filter(&filter))
            .boxed(),
    );

    if let Some(forward) = trnsys {
        layers.push(
            TrnSysLogLayer::new(forward)
                .with_filter(env_filter(&filter))
                .boxed(),
        );
    }

    Registry::default().with(layers)
}

/// Initializes tracing with custom layers and settings.
///
/// Called once, on the first call of the Type. Installs the global subscriber and the panic hook.
pub fn init_tracing() {
    let subscriber = build_subscriber(LogConfig::production());

    if let Err(err) = tracing::subscriber::set_global_default(subscriber) {
        diag(format_args!("cannot install the tracing subscriber: {err}"));
    }

    // panic hook
    std::panic::set_hook(Box::new(|panic_info| {
        error!("TrnSys Type Panicked: {:#}", panic_info);
    }));
}
