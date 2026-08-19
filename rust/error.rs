use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum NativeError {
    InvalidInput(String),
    OutOfMemory(String),
}

impl NativeError {
    pub(crate) fn out_of_memory(message: impl Into<String>) -> Self {
        Self::OutOfMemory(message.into())
    }

    pub(crate) fn is_out_of_memory(&self) -> bool {
        matches!(self, Self::OutOfMemory(_))
    }
}

impl Display for NativeError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        let (Self::InvalidInput(message) | Self::OutOfMemory(message)) = self;
        formatter.write_str(message)
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

        assert!(!invalid.is_out_of_memory());
        assert!(allocation.is_out_of_memory());
        assert_eq!(invalid.to_string(), allocation.to_string());
    }
}
