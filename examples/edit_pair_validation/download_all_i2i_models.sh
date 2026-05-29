#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_BASE_PATH="${MODEL_BASE_PATH:-./models}"
DOWNLOAD_SOURCE="${DOWNLOAD_SOURCE:-${DIFFSYNTH_DOWNLOAD_SOURCE:-modelscope}}"
IMAGE_DIR="${IMAGE_DIR:-data/edit_pair_validation/amazon_lipcare}"
SKIP_IMAGES="${SKIP_IMAGES:-0}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-${REPO_ROOT}/.hf_token}"

cd "${REPO_ROOT}"

export DIFFSYNTH_MODEL_BASE_PATH="${MODEL_BASE_PATH}"
export DIFFSYNTH_DOWNLOAD_SOURCE="${DOWNLOAD_SOURCE}"

if [[ -z "${HF_TOKEN:-}" && -f "${HF_TOKEN_FILE}" ]]; then
  HF_TOKEN="$(tr -d '\r\n' < "${HF_TOKEN_FILE}")"
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN
  export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN}}"
  echo "Hugging Face token detected from environment or ${HF_TOKEN_FILE}"
fi

if [[ "${SKIP_IMAGES}" != "1" ]]; then
  "${PYTHON_BIN}" examples/edit_pair_validation/all_i2i_reference_to_target.py prepare \
    --output-dir "${IMAGE_DIR}"
fi

"${PYTHON_BIN}" examples/edit_pair_validation/all_i2i_reference_to_target.py download-models \
  --download-source "${DOWNLOAD_SOURCE}" \
  --model-base-path "${MODEL_BASE_PATH}" \
  "$@"
