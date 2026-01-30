# Flow Nexus Swarm Skill

Use flow-nexus MCP tools for multi-agent swarm coordination, monitoring, and workflow orchestration.

## Available Tools (via flow-nexus MCP server)

### SWARM_OPS (3 tools)
- `swarm_create` - Create a new agent swarm with topology
- `swarm_status` - Get swarm status and agent details
- `swarm_control` - Control swarm (scale, pause, resume, stop)

### MONITOR (3 tools)
- `monitor_metrics` - Get performance metrics for agents/swarm
- `monitor_alerts` - Check and manage alert thresholds
- `monitor_health` - Health check for all system components

### WORKFLOW (1 tool)
- `workflow_execute` - Execute a multi-step agent workflow

## Usage Pattern

```bash
# Start the MCP server
npx flow-nexus@latest mcp start -m swarm

# Or use complete mode for all tools
npx flow-nexus@latest mcp start -m complete
```

## When to Use
- Coordinating multiple agents on complex tasks
- Monitoring swarm health and performance
- Executing multi-step workflows with agent handoffs
- Scaling agent count based on workload
