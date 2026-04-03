"""Custom IsaacLab environments registered as Gymnasium extensions.

Importing this package registers all custom envs with ``gymnasium``.
The import must happen *after* ``isaaclab_tasks`` has been imported
(which triggers the IsaacLab bootstrap and standard env registration).
"""

from p2p.isaaclab_envs import shadow_hand_pen  # noqa: F401
