#!/bin/bash
# Historical claude-flow swarm launch script.
# Preserved for archaeology while the repo migrates toward a thinner, repo-native
# swarm runner. Do not treat this as the canonical current execution path.
#
# Usage:
#   ./scripts/launch_swarms.sh           # Launch all swarms
#   ./scripts/launch_swarms.sh sota      # Launch SOTA pipeline swarm only
#   ./scripts/launch_swarms.sh youtube   # Launch YouTube artist swarm only
#   ./scripts/launch_swarms.sh training  # Launch training-inference swarm only
#   ./scripts/launch_swarms.sh status    # Show status
#   ./scripts/launch_swarms.sh stop      # Shutdown all swarms

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"
source "$SCRIPT_DIR/common_env.sh"
autovoice_activate_env

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

banner() {
    echo -e "${BLUE}"
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║     Claude-Flow Smart Swarm Orchestrator - AutoVoice      ║"
    echo "║                   Parallel Agent System                   ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

check_deps() {
    echo -e "${YELLOW}Checking dependencies...${NC}"

    # Check claude-flow
    if ! command -v claude-flow &> /dev/null; then
        echo -e "${RED}Error: claude-flow not found${NC}"
        echo "This legacy launcher requires the historical claude-flow stack."
        exit 1
    fi
    echo -e "  ${GREEN}✓${NC} claude-flow $(claude-flow --version 2>/dev/null | head -1)"

    # Check Python
    if [ ! -f "$PYTHON" ]; then
        echo -e "${RED}Error: Python not found at $PYTHON${NC}"
        exit 1
    fi
    echo -e "  ${GREEN}✓${NC} Python: $PYTHON"

    # Check config files
    if [ ! -f "$PROJECT_ROOT/config/swarm_config.yaml" ]; then
        echo -e "${RED}Error: Swarm config not found${NC}"
        exit 1
    fi
    echo -e "  ${GREEN}✓${NC} Swarm config: config/swarm_config.yaml"

    echo ""
}

launch_swarm() {
    local swarm=$1
    local parallel=${2:-10}

    echo -e "${GREEN}Launching swarm: ${swarm}${NC}"
    "$PYTHON" scripts/swarm_orchestrator.py \
        --swarm "$swarm" \
        --parallel "$parallel"
}

show_status() {
    echo -e "${BLUE}Swarm Status:${NC}"
    "$PYTHON" scripts/swarm_orchestrator.py --status

    # Also show hive-mind status
    echo ""
    echo -e "${BLUE}Hive Mind Status:${NC}"
    claude-flow hive-mind status 2>/dev/null || echo "  Hive mind not active"
}

stop_swarms() {
    echo -e "${YELLOW}Stopping all swarms...${NC}"
    "$PYTHON" scripts/swarm_orchestrator.py --shutdown
    claude-flow hive-mind shutdown 2>/dev/null || true
    echo -e "${GREEN}Swarms stopped${NC}"
}

# Main
banner
check_deps

case "${1:-all}" in
    all)
        echo -e "${GREEN}Launching ALL swarms (parallel)${NC}"
        echo ""

        # Initialize queen first
        echo -e "${YELLOW}Step 1: Initialize Queen Coordinator${NC}"
        "$PYTHON" scripts/swarm_orchestrator.py --init-only

        # Launch independent swarms in background
        echo ""
        echo -e "${YELLOW}Step 2: Launch Independent Swarms${NC}"

        # SOTA (P0 Critical) and YouTube (P1 Independent) can run in parallel
        launch_swarm "sota-dual-pipeline" &
        SOTA_PID=$!

        launch_swarm "youtube-artist" &
        YOUTUBE_PID=$!

        echo -e "${BLUE}SOTA Pipeline PID: $SOTA_PID${NC}"
        echo -e "${BLUE}YouTube Artist PID: $YOUTUBE_PID${NC}"

        # Wait for SOTA Phase 2 before starting training-inference
        echo ""
        echo -e "${YELLOW}Step 3: Training-Inference (after SOTA Phase 2)${NC}"
        echo "  Waiting for SOTA pipeline phase 2..."
        # In production, this would wait for beads task completion
        # For now, just note the dependency
        echo "  Note: training-inference depends on sota-dual-pipeline phase 2"
        echo "  It will be queued to start after SOTA completes phase 2"

        # Wait for background jobs
        wait

        echo ""
        echo -e "${GREEN}All swarms launched!${NC}"
        show_status
        ;;

    sota|sota-dual-pipeline)
        launch_swarm "sota-dual-pipeline" 10
        ;;

    youtube|youtube-artist)
        launch_swarm "youtube-artist" 4
        ;;

    training|training-inference)
        launch_swarm "training-inference" 5
        ;;

    status)
        show_status
        ;;

    stop|shutdown)
        stop_swarms
        ;;

    help|--help|-h)
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  all       Launch all swarms (default)"
        echo "  sota      Launch SOTA dual-pipeline swarm only"
        echo "  youtube   Launch YouTube artist training swarm only"
        echo "  training  Launch training-inference swarm only"
        echo "  status    Show swarm status"
        echo "  stop      Shutdown all swarms"
        echo "  help      Show this help"
        echo ""
        echo "Environment:"
        echo "  PYTHON    Path to Python interpreter (default: autovoice-thor conda)"
        ;;

    *)
        echo -e "${RED}Unknown command: $1${NC}"
        echo "Run '$0 help' for usage"
        exit 1
        ;;
esac
