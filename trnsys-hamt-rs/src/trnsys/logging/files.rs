//! One log file per TRNSYS unit.

use super::{UnitFn, diag, lock_bounded, log_file_name, unit_key};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tracing_subscriber::fmt::writer::MakeWriter;

/// Registry of the per-unit log files (`type_395_<unit>.log`).
///
/// A file is opened lazily, truncated, when its unit logs for the first time in this process.
/// Writes are unbuffered (one `write` per event) so that nothing is lost if TRNSYS kills the
/// process.
///
/// The unit is looked up when an event is written (`unit`: the TRNSYS kernel's *current* unit, a
/// global, so it is also right for events emitted from rayon worker threads).
pub(crate) struct UnitFiles {
    dir: PathBuf,
    unit: UnitFn,
    /// `None` marks a file that could not be opened: it is not retried for every event.
    files: Mutex<HashMap<i32, Option<File>>>,
}

impl UnitFiles {
    pub(crate) fn new(dir: PathBuf, unit: UnitFn) -> Self {
        UnitFiles {
            dir,
            unit,
            files: Mutex::new(HashMap::new()),
        }
    }

    /// Path of the log file of `unit`.
    pub(crate) fn path(&self, unit: i32) -> PathBuf {
        self.dir.join(log_file_name(unit))
    }

    /// Appends one formatted event to the file of the current unit.
    pub(crate) fn write_event(&self, event: &[u8]) -> io::Result<()> {
        let unit = unit_key((self.unit)());

        // Never wait forever inside logging code (see `lock_bounded`): drop the event instead.
        let Some(mut files) = lock_bounded(&self.files) else {
            return Ok(());
        };
        let slot = files
            .entry(unit)
            .or_insert_with(|| open_truncated(&self.path(unit)));

        match slot {
            Some(file) => file.write_all(event),
            None => Ok(()),
        }
    }
}

fn open_truncated(path: &Path) -> Option<File> {
    match OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(path)
    {
        Ok(file) => Some(file),
        Err(err) => {
            diag(format_args!(
                "cannot open log file {}: {err}",
                path.display()
            ));
            None
        }
    }
}

/// [`MakeWriter`] for the fmt layer: routes every event to the current unit's log file and,
/// optionally, to stdout.
pub(crate) struct FileSink {
    files: Arc<UnitFiles>,
    stdout: bool,
}

impl FileSink {
    pub(crate) fn new(files: Arc<UnitFiles>, stdout: bool) -> Self {
        FileSink { files, stdout }
    }
}

impl<'a> MakeWriter<'a> for FileSink {
    type Writer = FileSinkWriter<'a>;

    fn make_writer(&'a self) -> Self::Writer {
        FileSinkWriter { sink: self }
    }
}

/// The writer handed out by [`FileSink`]. The fmt layer passes a whole formatted event (one line
/// ending in `\n`) in a single `write_all`.
pub(crate) struct FileSinkWriter<'a> {
    sink: &'a FileSink,
}

impl Write for FileSinkWriter<'_> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let file_result = self.sink.files.write_event(buf);

        if self.sink.stdout {
            // A closed or broken stdout must not affect the log file.
            let _ = io::stdout().write_all(buf);
        }

        file_result.map(|()| buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        // Unbuffered: nothing to flush.
        Ok(())
    }
}
