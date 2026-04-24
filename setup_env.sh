#!/usr/bin/env bash
# setup_env.sh
# ─────────────────────────────────────────────────────────────────────────────
# One-command setup for the wildfire-uncertainty-gnn project.
# Run from the project root: bash setup_env.sh
#
# What this script does:
#   1. Creates the conda environment from environment.yml
#   2. Installs PyTorch Geometric scatter/sparse extensions (CUDA 11.8)
#   3. Installs the wildfire_gnn package in editable mode
#   4. Creates all data directory placeholders
#   5. Runs the environment validation notebook as a smoke test
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

ENV_NAME="wildfire-gnn"
PYTHON_VERSION="3.10"
CUDA_TAG="cu118"           # Change to cpu if no GPU
TORCH_VERSION="2.1.2"

# ── Colors ───────────────────────────────────────────────────────────────────
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
NC="\033[0m"

ok()   { echo -e "${GREEN}✓${NC}  $1"; }
warn() { echo -e "${YELLOW}!${NC}  $1"; }
err()  { echo -e "${RED}✗${NC}  $1"; exit 1; }

# ── 0. Check prerequisites ──────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════"
echo "  wildfire-gnn — Phase 0 Environment Setup"
echo "══════════════════════════════════════════════"
echo ""

command -v conda >/dev/null 2>&1 || err "conda not found. Install Miniconda first."
command -v git   >/dev/null 2>&1 || err "git not found."

ok "conda and git found"

# ── 1. Create / update conda environment ────────────────────────────────────
echo ""
echo "── Step 1: Creating conda environment '${ENV_NAME}' ──"

if conda info --envs | grep -q "^${ENV_NAME} "; then
    warn "Environment '${ENV_NAME}' already exists — updating"
    conda env update --name "${ENV_NAME}" --file environment.yml --prune
else
    conda env create --file environment.yml
fi

ok "Conda environment ready"

# ── 2. Activate and install PyG extensions ──────────────────────────────────
echo ""
echo "── Step 2: Installing PyTorch Geometric extensions ──"

# Use conda run to execute in the target env without activating
PYG_BASE="https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_TAG}.html"

conda run -n "${ENV_NAME}" pip install \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-spline-conv \
    -f "${PYG_BASE}" \
    --quiet

ok "PyG extensions installed"

# ── 3. Install wildfire_gnn in editable mode ────────────────────────────────
echo ""
echo "── Step 3: Installing wildfire_gnn package (editable) ──"

conda run -n "${ENV_NAME}" pip install -e . --quiet

ok "wildfire_gnn installed in editable mode"

# ── 4. Create directory structure and placeholder files ─────────────────────
echo ""
echo "── Step 4: Creating directory structure ──"

DIRS=(
    "data/raw"
    "data/interim/aligned"
    "data/processed"
    "data/features"
    "data/external"
    "reports/figures"
    "reports/tables"
    "checkpoints"
)

for d in "${DIRS[@]}"; do
    mkdir -p "${d}"
    touch "${d}/.gitkeep"
done

ok "Directory structure created"

# ── 5. Git setup ─────────────────────────────────────────────────────────────
echo ""
echo "── Step 5: Git setup ──"

if [ ! -d ".git" ]; then
    git init
    git add .
    git commit -m "chore: Phase 0 — initial project structure"
    ok "Git repo initialized with initial commit"
else
    warn "Git repo already initialized — skipping"
fi

# ── 6. Smoke test: run validation notebook ──────────────────────────────────
echo ""
echo "── Step 6: Running environment validation ──"

conda run -n "${ENV_NAME}" jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    --ExecutePreprocessor.timeout=120 \
    notebooks/00_environment_validation.ipynb 2>/dev/null \
    && ok "Validation notebook passed" \
    || warn "Validation notebook had errors — check notebooks/00_environment_validation.ipynb manually"

# ── Done ────────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════"
echo -e "${GREEN}  Phase 0 setup complete.${NC}"
echo "══════════════════════════════════════════════"
echo ""
echo "  Next steps:"
echo "    conda activate ${ENV_NAME}"
echo "    jupyter lab"
echo "    → open notebooks/00_environment_validation.ipynb"
echo ""