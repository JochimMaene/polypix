use std::fmt::{Display, Formatter};

/// Failure categories the Python boundary maps to distinct exception types.
///
/// Allocation messages are `&'static str` because every one of them is a fixed
/// literal: reporting that an allocation failed must not itself allocate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum NativeError {
    InvalidInput(String),
    OutOfMemory(&'static str),
}

/// The message every coverage-building allocation failure reports.
pub(crate) const COVERAGE_OUT_OF_MEMORY: &str = "Coverage result is too large to fit in memory.";

impl NativeError {
    pub(crate) fn out_of_memory(message: &'static str) -> Self {
        Self::OutOfMemory(message)
    }
}

impl Display for NativeError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput(message) => formatter.write_str(message),
            Self::OutOfMemory(message) => formatter.write_str(message),
        }
    }
}

impl From<String> for NativeError {
    fn from(message: String) -> Self {
        Self::InvalidInput(message)
    }
}

pub(crate) type NativeResult<T> = Result<T, NativeError>;

#[cfg(test)]
mod tests {
    use super::NativeError;

    #[test]
    fn category_does_not_depend_on_message_text() {
        let message = "Coverage result is too large to fit in memory.";
        let invalid = NativeError::InvalidInput(message.to_owned());
        let allocation = NativeError::out_of_memory(message);

        assert!(matches!(invalid, NativeError::InvalidInput(_)));
        assert!(matches!(allocation, NativeError::OutOfMemory(_)));
        assert_eq!(invalid.to_string(), allocation.to_string());
    }
}
