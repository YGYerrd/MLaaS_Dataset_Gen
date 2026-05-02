"""MLaaS service dataset generator package."""

from .services.runner import ServiceExecutionResult, execute_service, resolve_service_id

__all__ = ["ServiceExecutionResult", "execute_service", "resolve_service_id"]
