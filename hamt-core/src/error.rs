use strum::EnumString;
use thiserror::Error;

#[derive(strum::Display, Debug, EnumString)]
pub enum InvalidInputReason {
    #[strum(serialize = "missing value")]
    Missing,
    #[strum(serialize = "invalid format")]
    InvalidFormat,
    #[strum(serialize = "value out of range")]
    OutOfRange,
    #[strum(serialize = "duplicate value")]
    Duplicate,
    #[strum(to_string = "cannot parse to {0}")]
    ParseError(String),
    #[strum(to_string = "{0}")]
    Other(String),
}

#[derive(Error, Debug)]
pub enum Error {
    #[error("Invalid input '{1}' for '{0}': {2}")]
    InvalidInput(String, String, InvalidInputReason),

    #[error("Cannot lock '{0}'")]
    LockError(String),

    #[error("Unhandled error: {0:#}")]
    Other(#[source] Box<dyn std::error::Error + Send + Sync>),
}
