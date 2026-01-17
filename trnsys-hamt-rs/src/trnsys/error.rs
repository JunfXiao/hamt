use crate::trnsys::{found_bad_input, found_bad_parameter, Severity, TrnSysState};
use std::error::Error;
use std::io;
use thiserror::Error;
use tracing::error;

pub trait TrnSysErrorHandler {
    fn handle_in_trnsys(&self, state: &TrnSysState);
}

#[derive(Error, Debug)]
pub enum TrnSysError {
    #[error("Error in TrnSys Type: {0}")]
    GeneralError(String),
    #[error("Error in TrnSys: {0}")]
    InputError(#[from] InputError),
    #[error("IO Error: {0}")]
    IoError(#[from] io::Error),
    #[error("Cannot Unlock Variable: {0}")]
    LockError(String),
    #[error("Cannot convert {param}: {message}")]
    ConversionError { param: String, message: String },
    #[error("HAMT error: {0}")]
    HamtError(String),
    #[error("Unhandled error: {0}")]
    Other(#[source] Box<dyn Error + Send + Sync>),
}

impl TrnSysErrorHandler for TrnSysError {
    fn handle_in_trnsys(&self, state: &TrnSysState) {
        match self {
            TrnSysError::InputError(e) => {
                e.handle_in_trnsys(state);
            }
            TrnSysError::IoError(e) => {
                error!("IO error in TRNSYS type: {:?}", e);
            }
            TrnSysError::GeneralError(msg) => {
                error!("General TRNSYS error: {}", msg);
            }
            TrnSysError::LockError(var_name) => {
                error!("Lock error on variable: {}", var_name);
            }
            TrnSysError::ConversionError { param, message } => {
                error!("Conversion error on {}: {}", param, message);
            }
            TrnSysError::HamtError(msg) => {
                error!("HAMT error: {}", msg);
            }
            TrnSysError::Other(e) => {
                error!("Unhandled TRNSYS error: {:?}", e);
            }
        }
    }
}

impl TrnSysErrorHandler for anyhow::Error {
    fn handle_in_trnsys(&self, state: &TrnSysState) {
        if let Some(trnsys_err) = self.downcast_ref::<TrnSysError>() {
            trnsys_err.handle_in_trnsys(state);
        } else {
            error!("Unhandled Application Error: {:?}", self);
        }
    }
}

/// Input Error for TRNSYS
#[derive(Error, Debug)]
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

impl TrnSysErrorHandler for InputError {
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

impl From<String> for TrnSysError {
    fn from(value: String) -> Self {
        TrnSysError::GeneralError(value)
    }
}

impl From<Box<dyn Error + Send + Sync>> for TrnSysError {
    fn from(err: Box<dyn Error + Send + Sync>) -> Self {
        TrnSysError::Other(err)
    }
}
