use crate::trnsys::{Severity, TrnSysState, found_bad_input, found_bad_parameter};
use std::error::Error as StdError;
use std::io;
use thiserror::Error as ThisError;
use tracing::error;

pub trait ErrorHandler {
    fn handle_in_trnsys(&self, state: &TrnSysState);
}

#[derive(ThisError, Debug)]
pub enum Error {
    #[error("Error in TrnSys Type: {0}")]
    GeneralError(String),
    #[error("Error in TrnSys: {0:#}")]
    InputError(#[from] InputError),
    #[error("IO Error: {0}")]
    IoError(#[from] io::Error),
    #[error("Cannot Unlock Variable: {0}")]
    LockError(String),
    #[error("Cannot convert {param}: {message}")]
    ConversionError { param: String, message: String },
    #[error("HAMT error: {0}")]
    HamtError(String),
    #[error("Unhandled error: {0:#}")]
    Other(#[source] Box<dyn StdError + Send + Sync>),
    #[error("HAMT core error: {0}")]
    Core(#[from] hamt_core::error::Error),
}

impl ErrorHandler for Error {
    fn handle_in_trnsys(&self, state: &TrnSysState) {
        match self {
            Error::InputError(e) => {
                e.handle_in_trnsys(state);
            }
            Error::IoError(e) => {
                error!("IO error in TRNSYS type: {:#}", e);
            }
            Error::GeneralError(msg) => {
                error!("General TRNSYS error: {}", msg);
            }
            Error::LockError(var_name) => {
                error!("Lock error on variable: {}", var_name);
            }
            Error::ConversionError { param, message } => {
                error!("Conversion error on {}: {}", param, message);
            }
            Error::HamtError(msg) => {
                error!("HAMT error: {}", msg);
            }
            Error::Other(e) => {
                error!("Unhandled TRNSYS error: {:?}", e);
            }
            Error::Core(e) => {
                error!("HAMT core error: {:#}", e);
            }
        }
    }
}

impl ErrorHandler for anyhow::Error {
    fn handle_in_trnsys(&self, state: &TrnSysState) {
        if let Some(trnsys_err) = self.downcast_ref::<Error>() {
            trnsys_err.handle_in_trnsys(state);
        }
        error!("Application Error: {:#}", self);
    }
}

/// Input Error for TRNSYS
#[derive(ThisError, Debug)]
pub enum InputError {
    #[error("Bad Input at {index}: {message}")]
    BadInput { index: i32, message: String },
    #[error("Bad Parameter at {index}: {message}")]
    BadParameter { index: i32, message: String },
    #[error("Duplicate {item} found: {name}")]
    DuplicateItem { item: String, name: String },
    #[error("Missing {item}: {name}")]
    MissingItem { item: String, name: String },
    #[error("Wrong {item} definition in \"{name}\": {reason}")]
    WrongDefinition {
        item: String,
        name: String,
        reason: String,
    },
}

impl ErrorHandler for InputError {
    fn handle_in_trnsys(&self, state: &TrnSysState) {
        match self {
            InputError::BadInput { index, message } => {
                found_bad_input(index.clone(), Severity::Fatal, &format!("{:?}", self));
            }
            InputError::BadParameter { index, message } => {
                found_bad_parameter(index.clone(), Severity::Fatal, &format!("{:?}", self));
            }

            err => {
                found_bad_parameter(0, Severity::Fatal, &format!("{}", err));
            }
        }
    }
}

impl From<String> for Error {
    fn from(value: String) -> Self {
        Error::GeneralError(value)
    }
}

impl From<Box<dyn StdError + Send + Sync>> for Error {
    fn from(err: Box<dyn StdError + Send + Sync>) -> Self {
        Error::Other(err)
    }
}
