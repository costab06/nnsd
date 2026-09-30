#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY="$SCRIPT_DIR/src/nnsd/nnsd"

if [[ ! -x "$BINARY" ]]; then
    echo "nnsd binary not found. Building with CUDA support..."
    make -C "$SCRIPT_DIR/src/utils"
    make -C "$SCRIPT_DIR/src/nnsd" cuda
fi

# ── Defaults (override with environment variables) ────────────────────────────
MOUNT_POINT="${NNSD_MOUNT_POINT:-/opt/neural/data}"
CONFIG_FILE="${NNSD_CONFIG:-/opt/neural/conf/nnsd.conf}"
INTERFACE="${NNSD_INTERFACE:-}"
REST_PORT="${NNSD_REST_PORT:-9090}"

# ── Build argument list ───────────────────────────────────────────────────────
ARGS=(-config "$CONFIG_FILE" -mountPoint "$MOUNT_POINT" -restPort "$REST_PORT")

[[ -n "$INTERFACE" ]] && ARGS+=(-networkInterface "$INTERFACE")

# Optional debug flags — set env var to 1 to enable
[[ "${NNSD_DEBUG_PACKETS:-0}"   == "1" ]] && ARGS+=(-debugPackets)
[[ "${NNSD_DEBUG_MATH:-0}"      == "1" ]] && ARGS+=(-debugMath)
[[ "${NNSD_DEBUG_ENTER_EXIT:-0}" == "1" ]] && ARGS+=(-debugEnterExit)
[[ "${NNSD_DEBUG_WEIGHTS:-0}"   == "1" ]] && ARGS+=(-debugWeights)
[[ "${NNSD_FAILED_FACTS:-0}"    == "1" ]] && ARGS+=(-writeTrainingFailedFacts)

echo "Starting nnsd (foreground)"
echo "  Binary     : $BINARY"
echo "  Config     : $CONFIG_FILE"
echo "  Mount point: $MOUNT_POINT"
echo "  REST port  : $REST_PORT"
echo "  Interface  : ${INTERFACE:-(auto)}"
echo ""

exec "$BINARY" "${ARGS[@]}"
