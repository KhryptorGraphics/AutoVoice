# Flow Nexus Platform Skill

Use flow-nexus MCP tools for sandbox management, templates, storage, and system administration.

## Available Tools (via flow-nexus MCP server)

### SANDBOX (9 tools)
- `sandbox_create` - Create isolated execution environment
- `sandbox_list` - List active sandboxes
- `sandbox_exec` - Execute code in sandbox
- `sandbox_upload` - Upload files to sandbox
- `sandbox_download` - Download files from sandbox
- `sandbox_status` - Get sandbox status
- `sandbox_logs` - Get sandbox execution logs
- `sandbox_stop` - Stop a running sandbox
- `sandbox_delete` - Delete a sandbox

### TEMPLATES (3 tools)
- `template_list` - List available deployment templates
- `template_get` - Get template details and configuration
- `template_deploy` - Deploy from a template

### STORAGE (4 tools)
- `storage_put` - Store key-value data
- `storage_get` - Retrieve stored data
- `storage_list` - List stored keys
- `storage_delete` - Delete stored data

### SYSTEM (3 tools)
- `system_info` - Get system information
- `system_config` - View/update system configuration
- `system_reset` - Reset system state

### AUTH (12 tools)
- Authentication and user management tools

## Usage Pattern

```bash
# Start platform tools
npx flow-nexus@latest mcp start -m complete

# Or dev-focused subset
npx flow-nexus@latest mcp start -m dev
```

## When to Use
- Creating isolated sandboxes for code execution
- Managing deployment templates
- Persisting data across agent sessions
- System configuration and administration
