#!/bin/bash
# PUDA Training Launcher
# Usage: bash scripts/train_puda.sh [gpu_id]
#
# This script ensures stable long-running training for the PUDA architecture.
# Key features:
#   - Python unbuffered output
#   - Automatic GPU selection
#   - Error logging
#   - Resume capability

set -euo pipefail

# Configuration
GPU_ID="${1:-0}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT="${PROJECT_DIR}/scripts/19_puda.py"
LOG_DIR="${PROJECT_DIR}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/puda_${TIMESTAMP}.log"
PID_FILE="${LOG_DIR}/puda_${TIMESTAMP}.pid"

# Create log directory
mkdir -p "${LOG_DIR}"

# Environment setup
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TORCH_CUDNN_DETERMINISTIC=0

# Log header
{
    echo "========================================"
    echo "PUDA Training Launcher"
    echo "Timestamp: $(date)"
    echo "GPU_ID: ${GPU_ID}"
    echo "Project: ${PROJECT_DIR}"
    echo "Python: $(which python)"
    echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
    echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
    echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\")')"
    echo "========================================"
    echo ""
} | tee -a "${LOG_FILE}"

# Write PID
echo $$ > "${PID_FILE}"

# Run training with timeout and auto-retry
MAX_RETRIES=1
RETRY_COUNT=0

while [ ${RETRY_COUNT} -le ${MAX_RETRIES} ]; do
    echo "[$(date)] Starting PUDA training (attempt $((RETRY_COUNT + 1)))..." | tee -a "${LOG_FILE}"

    # Clean GPU memory before starting
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

    # Run the training script
    python -u "${SCRIPT}" 2>&1 | tee -a "${LOG_FILE}"

    EXIT_CODE=${PIPESTATUS[0]}

    if [ ${EXIT_CODE} -eq 0 ]; then
        echo "[$(date)] PUDA training completed successfully!" | tee -a "${LOG_FILE}"
        break
    elif [ ${EXIT_CODE} -eq 124 ]; then
        echo "[$(date)] Training timed out. Retrying..." | tee -a "${LOG_FILE}"
        RETRY_COUNT=$((RETRY_COUNT + 1))
    elif [ ${EXIT_CODE} -eq 137 ]; then
        echo "[$(date)] Training was killed (OOM?). Retrying..." | tee -a "${LOG_FILE}"
        RETRY_COUNT=$((RETRY_COUNT + 1))
    else
        echo "[$(date)] Training failed with exit code ${EXIT_CODE}. Check log for details." | tee -a "${LOG_FILE}"
        break
    fi

    if [ ${RETRY_COUNT} -gt ${MAX_RETRIES} ]; then
        echo "[$(date)] Max retries reached. Giving up." | tee -a "${LOG_FILE}"
    fi
done

# Clean up PID file
rm -f "${PID_FILE}"

echo "[$(date)] Log saved to: ${LOG_FILE}"
