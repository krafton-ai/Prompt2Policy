#!/usr/bin/env bash
# RLoop — One-command installer
# Usage: bash install.sh
set -euo pipefail

[ -t 0 ] || { echo "[FAIL] This script requires an interactive terminal." >&2; exit 1; }

# ──────────────────────────────────────────────────────────────
# Colors
# ──────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

info()  { printf "${CYAN}[INFO]${NC}  %s\n" "$*"; }
ok()    { printf "${GREEN}[OK]${NC}    %s\n" "$*"; }
warn()  { printf "${YELLOW}[WARN]${NC}  %s\n" "$*"; }
fail()  { printf "${RED}[FAIL]${NC}  %s\n" "$*"; exit 1; }

# ──────────────────────────────────────────────────────────────
# 1. Prerequisites
# ──────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

printf "\n${BOLD}=== RLoop Installer ===${NC}\n\n"

[ -f pyproject.toml ] || fail "pyproject.toml not found. Run this script from the repository root."

# Python 3.11+ — check versioned binaries first, then fall back to python3
info "Checking Python ..."
PYTHON_BIN=""
for candidate in python3.13 python3.12 python3.11 python3; do
    if command -v "$candidate" &>/dev/null; then
        PY_VER=$("$candidate" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
        PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
        if [ "$PY_MAJOR" -ge 3 ] && [ "$PY_MINOR" -ge 11 ]; then
            PYTHON_BIN="$candidate"
            break
        fi
    fi
done

if [ -n "$PYTHON_BIN" ]; then
    ok "Python $PY_VER ($PYTHON_BIN)"
else
    fail "Python 3.11+ required. Install python3.11 or newer."
fi

# uv
info "Checking uv ..."
if command -v uv &>/dev/null; then
    ok "uv $(uv --version 2>/dev/null | head -1)"
else
    fail "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
fi

# Node.js 18+ (for frontend)
install_node() {
    info "Installing Node.js 20 via NodeSource ..."
    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo bash -
    sudo apt-get install -y nodejs
    if command -v node &>/dev/null; then
        ok "Node.js $(node --version) installed."
    else
        fail "Node.js installation failed."
    fi
}

info "Checking Node.js ..."
NODE_OK=false
if command -v node &>/dev/null; then
    NODE_VER=$(node --version | sed 's/^v//')
    NODE_MAJOR=$(echo "$NODE_VER" | cut -d. -f1)
    if [ "$NODE_MAJOR" -ge 18 ]; then
        ok "Node.js v$NODE_VER"
        NODE_OK=true
    else
        NODE_REASON="Node.js v$NODE_VER is too old — frontend requires 18+."
    fi
else
    NODE_REASON="Node.js not found — frontend requires Node.js 18+."
fi

if [ "$NODE_OK" = false ]; then
    warn "$NODE_REASON"
    printf "  Install Node.js 20 now? [Y/n] "
    read -r INSTALL_NODE
    if [[ ! "$INSTALL_NODE" =~ ^[Nn]$ ]]; then
        install_node
        NODE_OK=true
    else
        fail "Node.js 18+ is required for the frontend."
    fi
fi

# GPU drivers
info "Checking GPU ..."
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) || true
    if [ -n "$GPU_NAME" ]; then
        ok "GPU: $GPU_NAME"
    else
        warn "nvidia-smi found but could not query GPU. Driver may not be loaded."
    fi
else
    warn "nvidia-smi not found — MuJoCo will use CPU rendering (slower)."
fi

# MuJoCo system packages (headless rendering)
# Ubuntu 24.04 renamed libegl1-mesa → libegl1, libgl1-mesa-glx → libgl1
info "Checking MuJoCo system packages ..."
MISSING_PKGS=""
for pkg in libegl1 libegl1-mesa libgl1 libgl1-mesa-glx libglu1-mesa; do
    dpkg -s "$pkg" &>/dev/null && continue
    # On Ubuntu 24.04 the -mesa variants don't exist; skip gracefully
    [[ "$pkg" == *-mesa* ]] && dpkg -s "${pkg%-mesa*}" &>/dev/null 2>&1 && continue
    [[ "$pkg" == *-mesa-glx ]] && dpkg -s "${pkg%-mesa-glx}" &>/dev/null 2>&1 && continue
    MISSING_PKGS="$MISSING_PKGS $pkg"
done
if [ -z "$MISSING_PKGS" ]; then
    ok "MuJoCo system packages installed."
else
    warn "Missing MuJoCo packages:$MISSING_PKGS"
    printf "  Install now? [Y/n] "
    read -r INSTALL_MUJOCO
    if [[ ! "$INSTALL_MUJOCO" =~ ^[Nn]$ ]]; then
        sudo apt-get install -y $MISSING_PKGS
        ok "MuJoCo system packages installed."
    else
        fail "MuJoCo packages are required for rendering. Install with: sudo apt-get install -y$MISSING_PKGS"
    fi
fi

# ──────────────────────────────────────────────────────────────
# 2. Python dependencies
# ──────────────────────────────────────────────────────────────
printf "\n${BOLD}--- Installing Python dependencies ---${NC}\n"
uv sync --all-extras
ok "Python dependencies installed."

# ──────────────────────────────────────────────────────────────
# 3. Frontend dependencies
# ──────────────────────────────────────────────────────────────
if [ "$NODE_OK" = true ] && [ -d "frontend" ]; then
    printf "\n${BOLD}--- Installing frontend dependencies ---${NC}\n"
    if ! (cd frontend && npm install); then
        fail "Frontend dependency installation failed. Check npm output above."
    fi
    ok "Frontend dependencies installed."
fi

# ──────────────────────────────────────────────────────────────
# 4. API key setup
# ──────────────────────────────────────────────────────────────
printf "\n${BOLD}--- API Key Configuration ---${NC}\n\n"

if [ -f .env ]; then
    info ".env file already exists."
    printf "  Overwrite with fresh configuration? [y/N] "
    read -r OVERWRITE
    if [[ ! "$OVERWRITE" =~ ^[Yy]$ ]]; then
        info "Keeping existing .env file."
        SKIP_ENV=true
    else
        SKIP_ENV=false
    fi
else
    SKIP_ENV=false
fi

if [ "$SKIP_ENV" = false ]; then
    # Start from the example template
    [ -f .env.example ] || fail ".env.example not found."
    cp .env.example .env

    # --- Anthropic ---
    printf "\n${BOLD}Anthropic API key${NC} (required — powers reward generation, judgment, revision)\n"
    printf "  Get yours at: https://console.anthropic.com/settings/keys\n"
    printf "  Key: "
    read -rs ANTHROPIC_KEY
    printf "\n"
    if [ -n "$ANTHROPIC_KEY" ]; then
        sed -i "s|^ANTHROPIC_API_KEY=.*|ANTHROPIC_API_KEY=$ANTHROPIC_KEY|" .env
        if grep -q "^ANTHROPIC_API_KEY=$ANTHROPIC_KEY" .env; then
            ok "Anthropic API key set."
        else
            warn "Failed to write ANTHROPIC_API_KEY to .env — please set it manually."
        fi
    else
        warn "Skipped — you can set ANTHROPIC_API_KEY in .env later."
    fi

    # --- Gemini ---
    printf "\n${BOLD}Google Gemini API key${NC} (recommended — VLM video-based behavior judgment)\n"
    printf "  Get yours at: https://aistudio.google.com/apikey\n"
    printf "  Key (Enter to skip): "
    read -rs GEMINI_KEY
    printf "\n"
    if [ -n "$GEMINI_KEY" ]; then
        sed -i "s|^GEMINI_API_KEY=.*|GEMINI_API_KEY=$GEMINI_KEY|" .env
        if grep -q "^GEMINI_API_KEY=$GEMINI_KEY" .env; then
            ok "Gemini API key set."
        else
            warn "Failed to write GEMINI_API_KEY to .env — please set it manually."
        fi
    else
        info "Skipped — VLM judgment will not be available until GEMINI_API_KEY is set."
    fi

    # --- OpenAI ---
    printf "\n${BOLD}OpenAI API key${NC} (optional — for GPT/o-series models)\n"
    printf "  Key (Enter to skip): "
    read -rs OPENAI_KEY
    printf "\n"
    if [ -n "$OPENAI_KEY" ]; then
        sed -i "s|^# OPENAI_API_KEY=.*|OPENAI_API_KEY=$OPENAI_KEY|" .env
        if grep -q "^OPENAI_API_KEY=$OPENAI_KEY" .env; then
            ok "OpenAI API key set."
        else
            warn "Failed to write OPENAI_API_KEY to .env — please set it manually."
        fi
    else
        info "Skipped."
    fi

    # --- MUJOCO_GL ---
    if command -v nvidia-smi &>/dev/null; then
        sed -i "s|^# MUJOCO_GL=egl|MUJOCO_GL=egl|" .env
        ok "MUJOCO_GL=egl enabled (headless GPU rendering)."
    fi

    ok ".env configured."
fi

# ──────────────────────────────────────────────────────────────
# 5. Summary
# ──────────────────────────────────────────────────────────────
printf "\n${BOLD}${GREEN}=== Installation complete ===${NC}\n\n"
printf "  ${BOLD}Verify:${NC}        uv run pytest tests/ -v\n"
printf "  ${BOLD}Start API:${NC}     uv run uvicorn p2p.api.app:app --host 0.0.0.0 --port 8000 --reload --reload-dir src\n"
printf "  ${BOLD}Start frontend:${NC} cd frontend && npm run dev\n"
printf "  ${BOLD}Open dashboard:${NC} http://localhost:3000\n"
printf "\n  Edit ${BOLD}.env${NC} to change API keys or model settings.\n"
printf "  See ${BOLD}.env.example${NC} for all available options.\n\n"
