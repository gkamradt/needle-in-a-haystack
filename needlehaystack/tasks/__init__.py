"""Tasks — the user-extension point.

Built-in tasks are registered on import. Third parties register their
own via `register_task("my_task", MyTaskClass)`.
"""

from __future__ import annotations

from .base import TASK_REGISTRY, Task, get_task, register_task, unregister_task
from .multi_needle import MultiNeedleTask
from .single_needle import SingleNeedleTask
from .uuid_chain import UuidChainTask
from .uuid_needle import UuidNeedleTask

# Built-in registrations. Anyone importing `needlehaystack.tasks` (or
# loading a config that names one of these) immediately has them
# available without a separate setup call.
register_task("single", SingleNeedleTask)
register_task("multi", MultiNeedleTask)
register_task("uuid", UuidNeedleTask)
register_task("uuid_chain", UuidChainTask)

__all__ = [
    "TASK_REGISTRY",
    "MultiNeedleTask",
    "SingleNeedleTask",
    "Task",
    "UuidChainTask",
    "UuidNeedleTask",
    "get_task",
    "register_task",
    "unregister_task",
]
