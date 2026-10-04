"""
ModelLab runtime management.

The runtime subsystem provides:

- controlled framework/runtime registry
- runtime compatibility checks
- isolated runtime discovery
- isolated runtime provisioning
- runtime-aware model resolution
"""

from .bridge import (
    RuntimeResolution,
    detect_runtime_requirement,
    resolve_model_with_runtime,
)
from .models import (
    RuntimeDefinition,
    RuntimeRequirement,
    RuntimeStatus,
)
from .provisioner import ProvisionResult, provision_runtime
from .registry import get_runtime, list_runtimes
from .resolver import resolve_runtime

__all__ = [
    "RuntimeDefinition",
    "RuntimeRequirement",
    "RuntimeStatus",
    "RuntimeResolution",
    "ProvisionResult",
    "get_runtime",
    "list_runtimes",
    "resolve_runtime",
    "provision_runtime",
    "detect_runtime_requirement",
    "resolve_model_with_runtime",
]
