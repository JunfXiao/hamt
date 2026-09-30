//! The debug flight recorder.
//!
//! DEBUG output is far too voluminous to write to disk all the time (a single debug-flooded run
//! produced a log of hundreds of MB), but it is exactly what is needed after something went wrong.
//! So every event at DEBUG and above is kept in a small in-memory ring per unit, and the rings are
//! written to `type_395_<unit>_debug.log` only when something goes wrong (see [`ErrorDumpLayer`],
//! [`RingSet::dump`]).

use super::format::LogTimer;
use super::{UnitFn, debug_dump_file_name, diag, lock_bounded, lock_bounded_for, unit_key};
use std::collections::{HashMap, VecDeque};
use std::fmt::{self, Write as _};
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tracing::field::{Field, Visit};
use tracing::{Event, Level, Subscriber};
use tracing_subscriber::fmt::writer::MakeWriter;
use tracing_subscriber::layer::{Context, Layer};

/// Environment variable with the size of the debug ring buffer of each unit, in MiB.
pub const ENV_RING_MB: &str = "TRNSYS_HAMT_RS_DEBUG_RING_MB";

const MIB: usize = 1024 * 1024;

/// Default ring size per unit [MiB].
pub(crate) const DEFAULT_RING_MB: usize = 8;

/// Longest `# reason:` text of a dump [characters].
const MAX_REASON_CHARS: usize = 300;

/// How long a dump waits for another dump that is writing the same files.
const DUMP_PATIENCE: Duration = Duration::from_secs(1);

/// The ring size derived from [`ENV_RING_MB`].
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct RingSetting {
    /// Byte budget of the ring of each unit; `None`: no ring at all (the layer is not installed, so
    /// DEBUG callsites stay disabled at zero cost).
    pub budget_bytes: Option<usize>,
    /// Set when the value was invalid; it is then replaced by the default. To be logged as a WARN
    /// once the subscriber is installed.
    pub warning: Option<String>,
}

/// Interprets the value of [`ENV_RING_MB`] (`None`: not set).
///
/// * unset or empty: the default, [`DEFAULT_RING_MB`] MiB;
/// * an integer number of MiB (surrounding whitespace is ignored): that size; `0` disables the ring;
/// * anything else (not an integer, negative, too large): the default, plus a warning.
pub(crate) fn parse_ring_setting(value: Option<&str>) -> RingSetting {
    let default = RingSetting {
        budget_bytes: Some(DEFAULT_RING_MB * MIB),
        warning: None,
    };

    let Some(raw) = value else {
        return default;
    };
    let text = raw.trim();
    if text.is_empty() {
        return default;
    }

    match text
        .parse::<usize>()
        .ok()
        .and_then(|mb| mb.checked_mul(MIB))
    {
        Some(0) => RingSetting {
            budget_bytes: None,
            warning: None,
        },
        Some(bytes) => RingSetting {
            budget_bytes: Some(bytes),
            warning: None,
        },
        None => RingSetting {
            warning: Some(format!(
                "ignoring invalid {ENV_RING_MB}={raw:?}: expected a whole number of MiB \
                 (0 disables the debug ring buffer); using the default of {DEFAULT_RING_MB} MiB"
            )),
            ..default
        },
    }
}

/// The formatted lines of one unit, oldest first, within a byte budget.
#[derive(Default)]
pub(super) struct Ring {
    lines: VecDeque<String>,
    /// Total length of `lines`.
    bytes: usize,
    /// Lines evicted since the start of the run.
    discarded: u64,
}

impl Ring {
    /// Appends `line` and evicts the oldest lines until the ring fits `budget` again. The newest
    /// line is always kept, even if it alone exceeds the budget.
    fn push(&mut self, line: String, budget: usize) {
        self.bytes = self.bytes.saturating_add(line.len());
        self.lines.push_back(line);

        while self.bytes > budget && self.lines.len() > 1 {
            let Some(oldest) = self.lines.pop_front() else {
                break;
            };
            self.bytes = self.bytes.saturating_sub(oldest.len());
            self.discarded = self.discarded.saturating_add(1);
        }
    }
}

/// The per-unit rings plus everything needed to dump them.
///
/// Nothing here may panic or block indefinitely: the rings are fed from every log call and dumped
/// from the panic hook (see [`lock_bounded`]).
pub(crate) struct RingSet {
    dir: PathBuf,
    /// The TRNSYS unit that is current when a line is recorded.
    unit: UnitFn,
    timer: LogTimer,
    /// Byte budget of each unit's ring.
    budget: usize,
    pub(super) rings: Mutex<HashMap<i32, Ring>>,
    /// Serializes dumps: two dumps must not write the same file at once.
    dump_gate: Mutex<()>,
    /// Set by every dump; see [`RingSet::dump_once`].
    dumped: AtomicBool,
}

impl RingSet {
    pub(crate) fn new(dir: PathBuf, unit: UnitFn, timer: LogTimer, budget: usize) -> Self {
        RingSet {
            dir,
            unit,
            timer,
            budget,
            rings: Mutex::new(HashMap::new()),
            dump_gate: Mutex::new(()),
            dumped: AtomicBool::new(false),
        }
    }

    /// Path of the dump of `unit`: `type_395_<unit>_debug.log`, next to its log file.
    pub(crate) fn dump_path(&self, unit: i32) -> PathBuf {
        self.dir.join(debug_dump_file_name(unit))
    }

    /// Files one formatted line under the unit that is current now.
    fn record(&self, line: String) {
        let unit = unit_key((self.unit)());
        let Some(mut rings) = lock_bounded(&self.rings) else {
            return;
        };
        rings.entry(unit).or_default().push(line, self.budget);
    }

    /// Units that currently have something to dump, in ascending order.
    fn units_with_entries(&self) -> Vec<i32> {
        let Some(rings) = lock_bounded(&self.rings) else {
            return Vec::new();
        };
        let mut units: Vec<i32> = rings
            .iter()
            .filter(|(_, ring)| !ring.lines.is_empty())
            .map(|(unit, _)| *unit)
            .collect();
        units.sort_unstable();
        units
    }

    /// The text of the dump file of `unit`: a header, then the entries in order.
    fn render(&self, unit: i32, reason: &str) -> Option<String> {
        let written = self.timer.now();

        let rings = lock_bounded(&self.rings)?;
        let ring = rings.get(&unit).filter(|ring| !ring.lines.is_empty())?;

        let mut text = String::with_capacity(ring.bytes.saturating_add(512));
        let _ = writeln!(text, "# trnsys-hamt-rs debug ring buffer dump");
        let _ = writeln!(text, "# reason: {reason}");
        let _ = writeln!(text, "# written: {written}");
        let _ = writeln!(
            text,
            "# unit: {unit}, entries: {} ({} bytes), older entries discarded: {}",
            ring.lines.len(),
            ring.bytes,
            ring.discarded
        );
        for line in &ring.lines {
            text.push_str(line);
            if !line.ends_with('\n') {
                text.push('\n');
            }
        }
        Some(text)
    }

    /// Writes `type_395_<unit>_debug.log` (overwriting it) for every unit whose ring is not empty.
    ///
    /// `reason` is reduced to its first line (at most [`MAX_REASON_CHARS`] characters). Failures are
    /// reported on stderr, never as a panic. Each ring is copied under its lock and written after the
    /// lock is released, so logging is not held up by file I/O.
    pub(crate) fn dump(&self, reason: &str) {
        // Waiting for a dump that is in progress is fine (they are short), waiting forever is not.
        let Some(_one_dump_at_a_time) = lock_bounded_for(&self.dump_gate, DUMP_PATIENCE) else {
            return;
        };
        self.dumped.store(true, Ordering::Release);

        let reason = first_line(reason);
        for unit in self.units_with_entries() {
            let Some(text) = self.render(unit, &reason) else {
                continue;
            };

            let path = self.dump_path(unit);
            if let Err(err) = std::fs::write(&path, text) {
                diag(format_args!("cannot write {}: {err}", path.display()));
            }
        }
    }

    /// Like [`dump`](Self::dump), but only if nothing has been dumped yet in the life of this
    /// `RingSet`: it does not repeat a dump, and in particular it does not replace the dump of an
    /// `ERROR` event, which was taken when the failure happened, names its cause, and could not
    /// have lost older lines to whatever was logged since.
    pub(crate) fn dump_once(&self, reason: &str) {
        if !self.dumped.swap(true, Ordering::AcqRel) {
            self.dump(reason);
        }
    }
}

/// The first line of `reason`, cut to [`MAX_REASON_CHARS`] characters (marked with `...`).
fn first_line(reason: &str) -> String {
    let line = reason.lines().next().unwrap_or("").trim_end();

    if line.chars().count() <= MAX_REASON_CHARS {
        line.to_string()
    } else {
        let mut cut: String = line.chars().take(MAX_REASON_CHARS).collect();
        cut.push_str("...");
        cut
    }
}

/// [`MakeWriter`] of the ring layer: every formatted event becomes one ring entry.
pub(crate) struct RingSink {
    rings: Arc<RingSet>,
}

impl RingSink {
    pub(crate) fn new(rings: Arc<RingSet>) -> Self {
        RingSink { rings }
    }
}

impl<'a> MakeWriter<'a> for RingSink {
    type Writer = RingWriter<'a>;

    fn make_writer(&'a self) -> Self::Writer {
        RingWriter {
            rings: &self.rings,
            event: Vec::new(),
        }
    }
}

/// Collects the text of one event and files it in the ring when it is dropped, i.e. when the fmt
/// layer is done with the event.
pub(crate) struct RingWriter<'a> {
    rings: &'a RingSet,
    event: Vec<u8>,
}

impl Write for RingWriter<'_> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.event.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl Drop for RingWriter<'_> {
    fn drop(&mut self) {
        if self.event.is_empty() {
            return;
        }

        let bytes = std::mem::take(&mut self.event);
        let line = match String::from_utf8(bytes) {
            Ok(line) => line,
            Err(invalid) => String::from_utf8_lossy(invalid.as_bytes()).into_owned(),
        };
        self.rings.record(line);
    }
}

/// Dumps the rings whenever an `ERROR` event is emitted.
///
/// It has to sit *after* the layers that record the event (ring, log file) and *before* the layer
/// that reports it to TRNSYS: inner layers see an event first, so the `ERROR` line is already in
/// the ring and in the log file, and the dump is on disk before TRNSYS (which turns every error
/// into a fatal one and may end the process) hears about it. Panics are covered too: the panic
/// hook logs an `error!`.
pub(crate) struct ErrorDumpLayer {
    rings: Arc<RingSet>,
}

impl ErrorDumpLayer {
    pub(crate) fn new(rings: Arc<RingSet>) -> Self {
        ErrorDumpLayer { rings }
    }
}

impl<S: Subscriber> Layer<S> for ErrorDumpLayer {
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let metadata = event.metadata();
        if *metadata.level() != Level::ERROR {
            return;
        }

        let mut message = MessageText::default();
        event.record(&mut message);
        let reason = format!(
            "ERROR event in {}: {}",
            metadata.target(),
            message
                .text
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
        );

        self.rings.dump(&reason);
    }
}

/// Extracts the `message` field of an event.
#[derive(Default)]
struct MessageText {
    text: String,
}

impl Visit for MessageText {
    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        if field.name() == "message" {
            let _ = write!(self.text, "{value:?}");
        }
    }
}
