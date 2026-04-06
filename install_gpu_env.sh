#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

REQ_FILE="${1:-/root/new_req.txt}"
VENV_DIR="${VENV_DIR:-/root/venv}"
OPENCV_VARIANT="${OPENCV_VARIANT:-headless}"   # headless or gui
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
MICROMAMBA_BIN="${MICROMAMBA_BIN:-/root/.local/bin/micromamba}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/root/.micromamba}"
BOOTSTRAP_PREFIX="${BOOTSTRAP_PREFIX:-/root/.gpu-bootstrap}"
TORCH_INDEX="https://download.pytorch.org/whl/cu121"
NVIDIA_INDEX="https://pypi.nvidia.com"
PIP_COMMON=(--no-cache-dir --prefer-binary --retries 20 --timeout 180)
TMP_DIR="$(mktemp -d)"
REQ_CLEAN="$TMP_DIR/requirements.clean.txt"
REQ_PYTORCH="$TMP_DIR/requirements.pytorch.txt"
REQ_REST="$TMP_DIR/requirements.rest.txt"
BUILD_LDFLAGS=""

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

cleanup() {
  rm -rf "$TMP_DIR"
}

on_err() {
  local line="$1"
  local exit_code="$2"
  printf 'ERROR: command failed at line %s (exit %s): %s\n' "$line" "$exit_code" "$BASH_COMMAND" >&2
  exit "$exit_code"
}

trap cleanup EXIT
trap 'on_err "$LINENO" "$?"' ERR

if [[ "${EUID}" -ne 0 ]]; then
  if command -v sudo >/dev/null 2>&1; then
    exec sudo -E bash "$0" "$@"
  fi
  die "Run this script as root (needed for ${VENV_DIR})"
fi

[[ -f "$REQ_FILE" ]] || die "Requirements file not found: $REQ_FILE"
[[ "$(uname -s)" == "Linux" ]] || die "This installer targets Linux"
case "$(uname -m)" in
  x86_64|amd64) ;;
  *) die "This installer targets Linux x86_64 because the pinned TensorRT wheels are Linux x86_64 packages." ;;
esac

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

pkg_version_from_req() {
  local pkg="$1"
  awk -F'==' -v pkg="$pkg" '$1==pkg {print $2}' "$REQ_CLEAN" | tail -n 1
}

normalize_requirements() {
  awk '
    {
      sub(/\r$/, "")
      if ($0 ~ /^[[:space:]]*$/) next
      if ($0 ~ /^[[:space:]]*#/) next
      print $0
    }
  ' "$REQ_FILE" > "$REQ_CLEAN"
}

download_to_stdout() {
  local url="$1"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- "$url"
  else
    die "Need curl or wget to bootstrap micromamba"
  fi
}

install_micromamba() {
  if [[ -x "$MICROMAMBA_BIN" ]]; then
    return 0
  fi

  require_cmd tar
  mkdir -p "$(dirname "$MICROMAMBA_BIN")"
  mkdir -p "$MAMBA_ROOT_PREFIX"

  local tmp_extract
  local url
  tmp_extract="$(mktemp -d)"
  url="https://micro.mamba.pm/api/micromamba/linux-64/latest"

  log "Bootstrapping micromamba"
  (
    cd "$tmp_extract"
    download_to_stdout "$url" | tar -xj -f - bin/micromamba
    mv "$tmp_extract/bin/micromamba" "$MICROMAMBA_BIN"
  )
  chmod +x "$MICROMAMBA_BIN"
  rm -rf "$tmp_extract"
}

bootstrap_python_and_toolchain() {
  install_micromamba

  local pkgs
  pkgs=(
    "python=${PYTHON_VERSION}"
    pip
    setuptools
    wheel
    c-compiler
    cxx-compiler
  )

  log "Creating self-contained Python/toolchain bootstrap prefix at $BOOTSTRAP_PREFIX"
  if [[ -d "$BOOTSTRAP_PREFIX/conda-meta" ]]; then
    MAMBA_ROOT_PREFIX="$MAMBA_ROOT_PREFIX" "$MICROMAMBA_BIN" install -y -p "$BOOTSTRAP_PREFIX" -c conda-forge "${pkgs[@]}"
  else
    MAMBA_ROOT_PREFIX="$MAMBA_ROOT_PREFIX" "$MICROMAMBA_BIN" create -y -p "$BOOTSTRAP_PREFIX" -c conda-forge "${pkgs[@]}"
  fi

  [[ -x "$BOOTSTRAP_PREFIX/bin/python" ]] || die "Bootstrap Python not found in $BOOTSTRAP_PREFIX"
  BUILD_LDFLAGS="-Wl,-rpath,${BOOTSTRAP_PREFIX}/lib"
}

run_in_build_env() {
  local merged_ldflags="${BUILD_LDFLAGS:-}"
  if [[ -n "${LDFLAGS:-}" ]]; then
    if [[ -n "$merged_ldflags" ]]; then
      merged_ldflags+=" ${LDFLAGS}"
    else
      merged_ldflags="${LDFLAGS}"
    fi
  fi

  if [[ -n "$merged_ldflags" ]]; then
    MAMBA_ROOT_PREFIX="$MAMBA_ROOT_PREFIX" "$MICROMAMBA_BIN" run -p "$BOOTSTRAP_PREFIX" env "LDFLAGS=$merged_ldflags" "$@"
  else
    MAMBA_ROOT_PREFIX="$MAMBA_ROOT_PREFIX" "$MICROMAMBA_BIN" run -p "$BOOTSTRAP_PREFIX" "$@"
  fi
}

create_venv() {
  log "Recreating virtual environment at $VENV_DIR"
  rm -rf "$VENV_DIR"
  run_in_build_env "$BOOTSTRAP_PREFIX/bin/python" -m venv --prompt venv "$VENV_DIR"
  [[ -x "$VENV_DIR/bin/python" ]] || die "Failed to create virtual environment at $VENV_DIR"
}

venv_pip() {
  run_in_build_env "$VENV_DIR/bin/python" -m pip "$@"
}

enforce_onnxruntime_gpu() {
  if [[ -n "$ONNXRUNTIME_GPU_VER" ]]; then
    log "Enforcing GPU ONNX Runtime"
    venv_pip uninstall -y onnxruntime onnxruntime-gpu >/dev/null 2>&1 || true
    venv_pip install "${PIP_COMMON[@]}" "onnxruntime-gpu==${ONNXRUNTIME_GPU_VER}"
  fi
}

require_cmd nvidia-smi
GPU_SUMMARY="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -n 1 || true)"
[[ -n "$GPU_SUMMARY" ]] || die "No NVIDIA GPU detected by nvidia-smi"
log "Using installed NVIDIA stack: $GPU_SUMMARY"

bootstrap_python_and_toolchain
create_venv

VENV_PY="$VENV_DIR/bin/python"

log "Upgrading pip/setuptools/wheel in the venv"
venv_pip install "${PIP_COMMON[@]}" --upgrade pip setuptools wheel

log "Adding pip-side build helpers"
venv_pip install "${PIP_COMMON[@]}" cmake ninja

normalize_requirements

grep -E '^(torch|torchvision|torchaudio)==' "$REQ_CLEAN" > "$REQ_PYTORCH" || true
if [[ ! -s "$REQ_PYTORCH" ]]; then
  die "Could not find torch/torchvision/torchaudio pins in $REQ_FILE"
fi

grep -Ev '^(torch|torchvision|torchaudio|onnxruntime-gpu|opencv-python|opencv-python-headless|insightface)==' "$REQ_CLEAN" > "$REQ_REST" || true

NUMPY_VER="$(pkg_version_from_req 'numpy' || true)"
CYTHON_VER="$(pkg_version_from_req 'Cython' || true)"
ONNXRUNTIME_GPU_VER="$(pkg_version_from_req 'onnxruntime-gpu' || true)"
INSIGHTFACE_VER="$(pkg_version_from_req 'insightface' || true)"
OPENCV_GUI_VER="$(pkg_version_from_req 'opencv-python' || true)"
OPENCV_HEADLESS_VER="$(pkg_version_from_req 'opencv-python-headless' || true)"

if [[ -n "$NUMPY_VER" || -n "$CYTHON_VER" ]]; then
  log "Bootstrapping NumPy/Cython before packages that build from source"
  PRE_PKGS=()
  [[ -n "$NUMPY_VER" ]] && PRE_PKGS+=("numpy==${NUMPY_VER}")
  [[ -n "$CYTHON_VER" ]] && PRE_PKGS+=("Cython==${CYTHON_VER}")
  venv_pip install "${PIP_COMMON[@]}" "${PRE_PKGS[@]}"
fi

log "Installing CUDA 12.1 PyTorch wheels from the official cu121 index"
venv_pip install "${PIP_COMMON[@]}" --extra-index-url "$TORCH_INDEX" -r "$REQ_PYTORCH"

if [[ -s "$REQ_REST" ]]; then
  log "Installing the remaining pinned requirements"
  venv_pip install "${PIP_COMMON[@]}" \
    --extra-index-url "$TORCH_INDEX" \
    --extra-index-url "$NVIDIA_INDEX" \
    -r "$REQ_REST"
fi

enforce_onnxruntime_gpu

case "$OPENCV_VARIANT" in
  headless)
    [[ -n "$OPENCV_HEADLESS_VER" ]] || die "OPENCV_VARIANT=headless requested, but opencv-python-headless is not pinned in $REQ_FILE"
    log "Installing OpenCV with headless as the active runtime flavor"
    if [[ -n "$OPENCV_GUI_VER" ]]; then
      venv_pip install "${PIP_COMMON[@]}" "opencv-python==${OPENCV_GUI_VER}"
    fi
    venv_pip install "${PIP_COMMON[@]}" "opencv-python-headless==${OPENCV_HEADLESS_VER}"
    ;;
  gui)
    [[ -n "$OPENCV_GUI_VER" ]] || die "OPENCV_VARIANT=gui requested, but opencv-python is not pinned in $REQ_FILE"
    log "Installing OpenCV with GUI as the active runtime flavor"
    if [[ -n "$OPENCV_HEADLESS_VER" ]]; then
      venv_pip install "${PIP_COMMON[@]}" "opencv-python-headless==${OPENCV_HEADLESS_VER}"
    fi
    venv_pip install "${PIP_COMMON[@]}" "opencv-python==${OPENCV_GUI_VER}"
    ;;
  *)
    die "Unsupported OPENCV_VARIANT=$OPENCV_VARIANT (use headless or gui)"
    ;;
esac

if [[ -n "$INSIGHTFACE_VER" ]]; then
  log "Installing InsightFace separately with the bootstrap compiler toolchain"
  venv_pip install "${PIP_COMMON[@]}" --no-deps "insightface==${INSIGHTFACE_VER}"

  log "Patching known insightface np.int usage for newer NumPy"
  run_in_build_env "$VENV_PY" - <<'PY'
import importlib.util
import pathlib
import re

spec = importlib.util.find_spec("insightface")
if spec is None or not spec.submodule_search_locations:
    print("insightface not installed; skipping patch")
    raise SystemExit(0)
root = pathlib.Path(next(iter(spec.submodule_search_locations)))
patched = 0
for py_file in root.rglob("*.py"):
    text = py_file.read_text(encoding="utf-8")
    updated = re.sub(r"\bnp\.int\b", "int", text)
    if updated != text:
        py_file.write_text(updated, encoding="utf-8")
        patched += 1
print(f"Patched {patched} insightface file(s)")
PY
fi

enforce_onnxruntime_gpu

log "Running pip integrity check"
run_in_build_env "$VENV_PY" -m pip check

log "Writing a frozen package snapshot"
run_in_build_env "$VENV_PY" -m pip freeze > "$VENV_DIR/installed-freeze.txt"

log "Running GPU verification"
VERIFY_JSON="$VENV_DIR/install-verification.json"
run_in_build_env "$VENV_PY" - <<'PY' | tee "$VERIFY_JSON"
import json

results = {}

import torch
results["torch_version"] = torch.__version__
results["torch_cuda_available"] = bool(torch.cuda.is_available())
if not results["torch_cuda_available"]:
    raise SystemExit("Torch CUDA is not available")
results["gpu_name"] = torch.cuda.get_device_name(0)
_ = (torch.randn((1024, 1024), device="cuda") @ torch.randn((1024, 1024), device="cuda")).sum().item()
results["torch_gpu_smoke_test"] = "ok"

import onnxruntime as ort
providers = ort.get_available_providers()
results["onnxruntime_version"] = ort.__version__
results["onnxruntime_providers"] = providers
if "CUDAExecutionProvider" not in providers:
    raise SystemExit(f"CUDAExecutionProvider missing from ONNX Runtime providers: {providers}")

import tensorrt as trt
results["tensorrt_version"] = trt.__version__
logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
if not builder:
    raise SystemExit("TensorRT builder creation failed")
results["tensorrt_builder"] = "ok"

import cv2
results["opencv_version"] = cv2.__version__

import insightface
results["insightface_version"] = getattr(insightface, "__version__", "unknown")

print(json.dumps(results, indent=2))
PY

cat > "$VENV_DIR/USAGE.txt" <<EOF2
Activation:
  source $VENV_DIR/bin/activate

Sanity check:
  python -c "import torch, onnxruntime as ort, tensorrt as trt; print(torch.cuda.get_device_name(0)); print(ort.get_available_providers()); print(trt.__version__)"

Notes:
  * A self-contained bootstrap prefix is kept at: $BOOTSTRAP_PREFIX
  * It provides the Python/toolchain used to build insightface cleanly without relying on the host package manager.
EOF2

log "Done. Activate with: source $VENV_DIR/bin/activate"
log "Verification summary saved to: $VERIFY_JSON"
