"""Root conftest — guards against isaacsim bootstrap crash.

The ``isaacsim`` package calls ``bootstrap_kernel()`` at import time
(``isaacsim/__init__.py`` line 220), which crashes pytest because it
tries to read stdin while pytest captures output.  We insert a
lightweight stub into ``sys.modules`` so that pytest's plugin discovery
never triggers the real import.  The stub is transparent to our code
because all IsaacLab imports in ``p2p`` are lazy (inside functions),
so they replace the stub with the real module when actually needed.
"""

from __future__ import annotations

import sys
import types

if "isaacsim" not in sys.modules:
    _stub = types.ModuleType("isaacsim")
    _stub.__path__ = []  # make it a proper package for submodule imports
    sys.modules["isaacsim"] = _stub
