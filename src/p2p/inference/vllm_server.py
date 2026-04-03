"""vLLM server lifecycle management.

Starts/stops vLLM as a subprocess for native video input to Qwen3.5-27B.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import time

import requests

from p2p.settings import VLLM_HOST, VLLM_MODEL, VLLM_PORT

logger = logging.getLogger(__name__)


def start_vllm_server(
    model: str = VLLM_MODEL,
    port: int = VLLM_PORT,
    max_model_len: int = 32768,
    timeout: int = 300,
    allowed_media_path: str | None = None,
) -> subprocess.Popen:
    """Launch vLLM as a subprocess and wait until healthy.

    Args:
        model: HuggingFace model ID (e.g. "Qwen/Qwen3.5-27B").
        port: Port for the OpenAI-compatible API server.
        max_model_len: Maximum context length (lower = less VRAM, faster startup).
        timeout: Seconds to wait for the server to become healthy.
        allowed_media_path: Directory for local file:// video access.

    Returns:
        The running subprocess handle.

    Raises:
        TimeoutError: If the server doesn't become healthy within *timeout*.
        RuntimeError: If the subprocess exits before becoming healthy.
    """
    # Use the conda env's Python directly (avoids conda run issues).
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    python_bin = os.path.join(conda_prefix, "bin", "python") if conda_prefix else "python"

    cmd = [
        python_bin,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--tensor-parallel-size",
        "1",
        "--max-model-len",
        str(max_model_len),
        "--reasoning-parser",
        "qwen3",
        # Qwen3.5 uses hybrid attention+Mamba layers; CUDA graph capture
        # has a bug in the nightly build's causal_conv1d, so skip it.
        "--enforce-eager",
    ]
    if allowed_media_path:
        cmd.extend(["--allowed-local-media-path", allowed_media_path])
    # Ensure conda env's libstdc++ is on LD_LIBRARY_PATH (has CXXABI_1.3.15
    # which the nightly vLLM build requires but the system lib may lack).
    env = os.environ.copy()
    conda_lib = os.path.join(conda_prefix, "lib") if conda_prefix else ""
    if conda_lib and os.path.isdir(conda_lib):
        env["LD_LIBRARY_PATH"] = f"{conda_lib}:{env.get('LD_LIBRARY_PATH', '')}"

    log_path = os.path.join(tempfile.gettempdir(), f"vllm_{port}.log")
    log_file = open(log_path, "a")  # noqa: SIM115
    logger.info("Starting vLLM: %s (log: %s)", " ".join(cmd), log_path)
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)
    # Child process inherited the fd; close parent's copy to avoid leak.
    log_file.close()

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            msg = f"vLLM exited with code {proc.returncode} before becoming healthy"
            raise RuntimeError(msg)
        if vllm_health_check(port):
            logger.info("vLLM healthy on port %d", port)
            return proc
        time.sleep(2)

    proc.terminate()
    proc.wait(timeout=10)
    msg = f"vLLM failed to become healthy within {timeout}s"
    raise TimeoutError(msg)


def stop_vllm_server(proc: subprocess.Popen) -> None:
    """Gracefully stop a vLLM subprocess."""
    if proc.poll() is not None:
        return
    logger.info("Stopping vLLM (pid=%d)", proc.pid)
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        logger.warning("vLLM didn't stop gracefully, killing")
        proc.kill()
        proc.wait()


def vllm_health_check(host: str = VLLM_HOST, port: int = VLLM_PORT) -> bool:
    """Check if vLLM server is healthy on the given host:port."""
    # For health checks, use localhost if the server is local (bound to 0.0.0.0)
    check_host = "localhost" if host == "0.0.0.0" else host
    try:
        resp = requests.get(f"http://{check_host}:{port}/health", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False
