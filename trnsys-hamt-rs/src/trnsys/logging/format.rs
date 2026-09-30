//! Event formatting shared by every sink (unit log files, stdout, debug ring buffer).

use super::{TimeFn, UnitFn};
use std::fmt;
use std::sync::Arc;
use tracing::{Event, Subscriber};
use tracing_subscriber::fmt::format::{self, Full, Writer};
use tracing_subscriber::fmt::time::{FormatTime, OffsetTime, UtcTime};
use tracing_subscriber::fmt::{FmtContext, FormatEvent, FormatFields};
use tracing_subscriber::registry::LookupSpan;

/// Timestamp source for log lines (RFC 3339, see [`LogTimer::local`]).
///
/// A thin, cheaply clonable wrapper so that the very same timer can be handed to the file format,
/// to the ring-buffer format and to the header of a ring-buffer dump, and so that unit tests can
/// substitute a deterministic one.
#[derive(Clone)]
pub(crate) struct LogTimer(Arc<dyn Fn(&mut Writer<'_>) -> fmt::Result + Send + Sync>);

impl LogTimer {
    /// Wraps an arbitrary timestamp writer.
    pub(crate) fn new(
        write_timestamp: impl Fn(&mut Writer<'_>) -> fmt::Result + Send + Sync + 'static,
    ) -> Self {
        LogTimer(Arc::new(write_timestamp))
    }

    /// Local time as RFC 3339 (e.g. `2026-09-30T14:03:11.123456+02:00`).
    ///
    /// The UTC offset is determined once, here. If the system cannot tell the local offset the
    /// timer falls back to UTC instead of failing: logging must never prevent a simulation from
    /// starting.
    pub(crate) fn local() -> Self {
        match OffsetTime::local_rfc_3339() {
            Ok(timer) => LogTimer::new(move |w| timer.format_time(w)),
            Err(_) => {
                let timer = UtcTime::rfc_3339();
                LogTimer::new(move |w| timer.format_time(w))
            }
        }
    }

    /// The current time as a string (no trailing space).
    pub(crate) fn now(&self) -> String {
        let mut out = String::new();
        let formatted = (self.0)(&mut Writer::new(&mut out));
        if formatted.is_err() {
            out = String::from("<unknown time>");
        }
        out
    }
}

impl FormatTime for LogTimer {
    fn format_time(&self, w: &mut Writer<'_>) -> fmt::Result {
        (self.0)(w)
    }
}

/// Prepends `[Unit N] [T=t] ` (current TRNSYS unit and simulation time in hours) to the output of
/// the wrapped event formatter.
///
/// The unit and the time are read through the injected providers when the event is formatted, so
/// the prefix is always that of the unit that is current at that moment. In the DLL the providers
/// read the TRNSYS kernel; unit tests inject their own.
pub(crate) struct UnitNoFmt<F> {
    inner: F,
    unit: UnitFn,
    sim_time: TimeFn,
}

impl<F> UnitNoFmt<F> {
    pub(crate) fn new(inner: F, unit: UnitFn, sim_time: TimeFn) -> Self {
        UnitNoFmt {
            inner,
            unit,
            sim_time,
        }
    }
}

impl<S, N, F> FormatEvent<S, N> for UnitNoFmt<F>
where
    S: Subscriber + for<'a> LookupSpan<'a>,
    N: for<'writer> FormatFields<'writer> + 'static,
    F: FormatEvent<S, N>,
{
    fn format_event(
        &self,
        ctx: &FmtContext<'_, S, N>,
        mut writer: Writer<'_>,
        event: &Event<'_>,
    ) -> fmt::Result {
        write!(
            writer,
            "[Unit {}] [T={}] ",
            (self.unit)(),
            (self.sim_time)()
        )?;

        self.inner.format_event(ctx, writer.by_ref(), event)
    }
}

/// The event formatter used by every sink: local-time timestamp, level, target, `file:line`, no
/// ANSI escapes, behind the `[Unit N] [T=t]` prefix.
///
/// The settings are applied to the [`format::Format`] that is wrapped in [`UnitNoFmt`], i.e. to the
/// formatter that really renders the events. (`fmt::Layer::with_timer` / `with_file` / ... only
/// configure the layer's *default* formatter, which `event_format(..)` throws away.)
pub(crate) fn event_format(
    timer: &LogTimer,
    unit: &UnitFn,
    sim_time: &TimeFn,
) -> UnitNoFmt<format::Format<Full, LogTimer>> {
    let inner = tracing_subscriber::fmt::format()
        .with_timer(timer.clone())
        .with_file(true)
        .with_line_number(true)
        .with_ansi(false);

    UnitNoFmt::new(inner, unit.clone(), sim_time.clone())
}
