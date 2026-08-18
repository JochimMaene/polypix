use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum NativeError {
    InvalidInput(String),
    Materialization(String),
}

impl NativeError {
    pub(crate) fn materialization(message: impl Into<String>) -> Self {
        Self::Materialization(message.into())
    }

    pub(crate) fn is_materialization(&self) -> bool {
        matches!(self, Self::Materialization(_))
    }

    pub(crate) fn message(&self) -> &str {
        match self {
            Self::InvalidInput(message) | Self::Materialization(message) => message,
        }
    }
}

impl Display for NativeError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.message())
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
        let message = "Coverage result is too large to materialize.";
        let invalid = NativeError::InvalidInput(message.to_owned());
        let allocation = NativeError::materialization(message);

        assert!(!invalid.is_materialization());
        assert!(allocation.is_materialization());
        assert_eq!(invalid.message(), allocation.message());
    }
}
