"""Versioned task semantics shared by generation and composition."""
from .contracts import TaskContract, TaskContractError, resolve_task_contract

__all__ = ["TaskContract", "TaskContractError", "resolve_task_contract"]
