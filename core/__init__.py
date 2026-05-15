"""Core package: configuration, security taxonomy, and exception hierarchy."""

from core.config import Settings, get_settings
from core.exceptions import (
    ConfigurationError,
    DocumentProcessingError,
    InfrastructureError,
    IngestionError,
    LLMConnectionError,
    LLMError,
    LLMGenerationError,
    LLMResponseFormatError,
    LexicalIndexError,
    MetadataValidationError,
    PipelineError,
    RoutingError,
    SecurityError,
    UnauthorizedAccessError,
    UserNotFoundError,
    VectorDBConnectionError,
    ZeroTrustBaseError,
)
from core.security import (
    ALLOWED_DEPARTMENT_WILDCARD,
    AllowedDepartment,
    ClearanceLevel,
    Department,
)

__all__: list[str] = [
    # Config
    "Settings",
    "get_settings",
    # Security taxonomy
    "ALLOWED_DEPARTMENT_WILDCARD",
    "AllowedDepartment",
    "ClearanceLevel",
    "Department",
    # Exceptions — root
    "ZeroTrustBaseError",
    # Exceptions — configuration
    "ConfigurationError",
    # Exceptions — security
    "SecurityError",
    "UnauthorizedAccessError",
    "MetadataValidationError",
    "UserNotFoundError",
    # Exceptions — infrastructure
    "InfrastructureError",
    "VectorDBConnectionError",
    "LexicalIndexError",
    "LLMError",
    "LLMConnectionError",
    "LLMGenerationError",
    "LLMResponseFormatError",
    # Exceptions — pipeline
    "PipelineError",
    "DocumentProcessingError",
    "IngestionError",
    "RoutingError",
]
