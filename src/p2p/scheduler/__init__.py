"""Job scheduler with pluggable execution backends (local, SSH, etc.).

This module provides manifest-based, stateless job orchestration:

  - **SSH tab**: always uses the scheduler (remote + local job dispatch).
  - **E2E tab**: always routes through the scheduler for local execution.
  - **Benchmark tab**: still uses the legacy pipeline (migration pending).

Both paths write to the same ``runs/<session_id>/`` directory structure,
so read endpoints (GET /sessions, GET /sessions/{id}) work transparently
regardless of which pipeline started the session.
"""
