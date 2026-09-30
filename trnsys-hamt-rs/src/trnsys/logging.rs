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
//! * A debug flight recorder: every event at DEBUG and above is also kept in an in-memory ring per
//!   unit (8 MiB by default), and `type_395_<unit>_debug.log` is written from it when an `ERROR` is
//!   logged (which includes panics), or when the simulation ends with an error reported by another
//!   component. See [`ring`].
//!
//! The level of the log file and of the TRNSYS forwarding is controlled by the `TRNSYS_HAMT_RS`
//! environment variable (`tracing` `EnvFilter` directives, default `info`), the size of the debug
//! rings by `TRNSYS_HAMT_RS_DEBUG_RING_MB`. Every layer has its own filter (there is no global one),
//! so the rings, which need DEBUG, do not change what the other layers receive.
//!
//! Everything that does not need the TRNSYS kernel lives behind [`build_subscriber`], which takes
//! the current unit and the simulation time as injected providers, so it can be unit-tested
//! without TRNSYS. [`init_tracing`] wires it to the kernel.

mod files;
mod format;
mod ring;
#[cfg(test)]
mod tests;

use crate::TYPE_NUMBER;
use crate::trnsys::{Severity, get_current_unit, get_simulation_time, log_message};
use files::{FileSink, UnitFiles};
use format::{LogTimer, event_format};
use ring::{ErrorDumpLayer, RingSet, RingSink, parse_ring_setting};
use std::fmt::{self, Write as _};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, TryLockError};
use std::time::{Duration, Instant};
use tracing::field::{Field, Visit};
use tracing::{Event, Level, Subscriber, error};
use tracing_subscriber::filter::{EnvFilter, LevelFilter};
use tracing_subscriber::layer::{Context, Layer, SubscriberExt};
use tracing_subscriber::registry::Registry;

pub use ring::ENV_RING_MB;

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
    /// Byte budget of the debug ring buffer of each unit (`None`: no ring buffer, and no
    /// `type_395_<unit>_debug.log`).
    pub ring_bytes: Option<usize>,
}

impl LogConfig {
    /// The configuration of the DLL: working directory, TRNSYS kernel providers, environment
    /// variables, stdout and TRNSYS forwarding on, and a debug ring of `ring_bytes` per unit.
    pub(crate) fn production(ring_bytes: Option<usize>) -> Self {
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
            ring_bytes,
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

/// `type_395_<unit>_debug.log`: the dump of the debug ring buffer, next to the log file.
pub(crate) fn debug_dump_file_name(unit: i32) -> String {
    format!("type_{}_{}_debug.log", TYPE_NUMBER, unit_key(unit))
}

/// Best-effort diagnostic on stderr for problems of the logging machinery itself. Never panics
/// (a closed or failing stderr is ignored), never logs through `tracing`.
pub(crate) fn diag(message: fmt::Arguments<'_>) {
    use std::io::Write;
    let _ = writeln!(std::io::stderr(), "[trnsys-hamt-rs] {message}");
}

/// How long logging code waits for a contended lock before it gives up on one record.
const LOCK_PATIENCE: Duration = Duration::from_millis(50);

/// Locks `mutex` for logging code, which must never deadlock or panic: [`lock_bounded_for`] with
/// the default patience ([`LOCK_PATIENCE`]).
pub(crate) fn lock_bounded<T>(mutex: &Mutex<T>) -> Option<MutexGuard<'_, T>> {
    lock_bounded_for(mutex, LOCK_PATIENCE)
}

/// Locks `mutex` for logging code, which must never deadlock or panic.
///
/// * A poisoned mutex is recovered (`into_inner`): a panic elsewhere must not silence logging,
///   and the panic hook itself logs.
/// * A mutex that stays locked for `patience` yields `None` (the caller drops the record) instead
///   of blocking. This covers a panic that unwinds through logging code while it holds the lock:
///   the panic hook then logs from the very thread that owns the lock.
pub(crate) fn lock_bounded_for<T>(
    mutex: &Mutex<T>,
    patience: Duration,
) -> Option<MutexGuard<'_, T>> {
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
                if started.elapsed() >= patience {
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

/// The handle on the debug rings of a subscriber built by [`build_subscriber`] (no-op if the ring
/// buffer is disabled).
#[derive(Clone)]
pub(crate) struct LogHandle {
    rings: Option<Arc<RingSet>>,
}

impl LogHandle {
    /// Writes `type_395_<unit>_debug.log` for every unit with a non-empty ring; see
    /// [`dump_debug_rings`].
    pub(crate) fn dump_debug_rings(&self, reason: &str) {
        if let Some(rings) = &self.rings {
            rings.dump(reason);
        }
    }

    /// Like [`dump_debug_rings`](Self::dump_debug_rings), but only if nothing has been dumped yet.
    pub(crate) fn dump_debug_rings_once(&self, reason: &str) {
        if let Some(rings) = &self.rings {
            rings.dump_once(reason);
        }
    }
}

/// Builds the subscriber of the DLL from `config`; it touches no global state.
///
/// Every layer carries its own filter (no global filter), so a layer never sees events that were
/// not meant for it. In this order (an event reaches the layers in this order, inner ones first):
///
/// 1. the debug ring buffer: DEBUG and above, whatever `TRNSYS_HAMT_RS` says (not installed if the
///    ring is disabled, which keeps the DEBUG callsites disabled at zero cost),
/// 2. the unit log file (and stdout), filtered by `TRNSYS_HAMT_RS`,
/// 3. the dump of the debug rings on `ERROR` events: after the ring and the log file, so both
///    already hold the `ERROR` line, and before TRNSYS is told about it,
/// 4. the forwarding to TRNSYS, filtered by `TRNSYS_HAMT_RS` and then by its own threshold.
///
/// Also returns a [`LogHandle`] to dump the rings by hand.
pub(crate) fn build_subscriber(
    config: LogConfig,
) -> (impl Subscriber + Send + Sync + 'static, LogHandle) {
    let LogConfig {
        dir,
        unit,
        sim_time,
        timer,
        filter,
        stdout,
        trnsys,
        ring_bytes,
    } = config;

    let files = Arc::new(UnitFiles::new(dir.clone(), unit.clone()));
    let rings =
        ring_bytes.map(|budget| Arc::new(RingSet::new(dir, unit.clone(), timer.clone(), budget)));

    let mut layers: Vec<BoxedLayer> = Vec::new();

    if let Some(rings) = &rings {
        layers.push(
            tracing_subscriber::fmt::layer::<Registry>()
                .with_writer(RingSink::new(rings.clone()))
                .with_ansi(false)
                .event_format(event_format(&timer, &unit, &sim_time))
                .with_filter(LevelFilter::DEBUG)
                .boxed(),
        );
    }

    layers.push(
        tracing_subscriber::fmt::layer::<Registry>()
            .with_writer(FileSink::new(files, stdout))
            .with_ansi(false)
            .event_format(event_format(&timer, &unit, &sim_time))
            .with_filter(env_filter(&filter))
            .boxed(),
    );

    if let Some(rings) = &rings {
        layers.push(
            ErrorDumpLayer::new(rings.clone())
                .with_filter(LevelFilter::ERROR)
                .boxed(),
        );
    }

    if let Some(forward) = trnsys {
        layers.push(
            TrnSysLogLayer::new(forward)
                .with_filter(env_filter(&filter))
                .boxed(),
        );
    }

    (Registry::default().with(layers), LogHandle { rings })
}

/// The handle of the subscriber installed by [`init_tracing`].
static LOG_HANDLE: OnceLock<LogHandle> = OnceLock::new();

/// Writes the debug ring buffers to disk: for every unit with a non-empty ring,
/// `type_395_<unit>_debug.log` (next to its log file, overwritten) with a header (`reason`, time,
/// unit, number of entries and of discarded older entries) followed by the last DEBUG-and-above
/// lines of that unit.
///
/// Happens automatically on every `ERROR` event (which includes panics); this function is for
/// other triggers. `reason` is reduced to its first line. Does nothing before [`init_tracing`] or if
/// the ring buffer is disabled. Never panics and never blocks indefinitely, so it is safe to call
/// from a panic hook.
pub fn dump_debug_rings(reason: &str) {
    if let Some(handle) = LOG_HANDLE.get() {
        handle.dump_debug_rings(reason);
    }
}

/// Like [`dump_debug_rings`], but only if the rings have not been dumped yet in the life of the
/// process (neither by this function, nor by an `ERROR` event, nor by [`dump_debug_rings`]): a
/// dump of the failure itself, which names its cause, must not be replaced by a later, vaguer one.
pub fn dump_debug_rings_once(reason: &str) {
    if let Some(handle) = LOG_HANDLE.get() {
        handle.dump_debug_rings_once(reason);
    }
}

/// Initializes tracing with custom layers and settings.
///
/// Called once, on the first call of the Type. Installs the global subscriber and the panic hook.
pub fn init_tracing() {
    let ring = parse_ring_setting(
        std::env::var_os(ENV_RING_MB)
            .map(|value| value.to_string_lossy().into_owned())
            .as_deref(),
    );

    let (subscriber, handle) = build_subscriber(LogConfig::production(ring.budget_bytes));

    if let Err(err) = tracing::subscriber::set_global_default(subscriber) {
        diag(format_args!("cannot install the tracing subscriber: {err}"));
    }
    let _ = LOG_HANDLE.set(handle);

    if let Some(warning) = &ring.warning {
        tracing::warn!("{warning}");
    }

    // panic hook: the `error!` reaches the log, TRNSYS and (via the ERROR trigger) the debug dump
    std::panic::set_hook(Box::new(|panic_info| {
        error!("TrnSys Type Panicked: {:#}", panic_info);
    }));
}
