# Augment MCP Quick Reference

## Installation

```bash
# Run the setup script
./scripts/setup_augment_mcp.sh

# Or install manually
npm install -g claude-flow@alpha ruv-swarm flow-nexus@latest
```

## Configuration Files

- **`.vscode/mcp.json`** - MCP server definitions
- **`.vscode/settings.json`** - Augment VSCode settings
- **`.claude/settings.json`** - Claude Code hooks and permissions

## Available MCP Servers

| Server | Purpose | Command |
|--------|---------|---------|
| **claude-flow** | SPARC methodology, agent orchestration | `npx claude-flow@alpha mcp start` |
| **ruv-swarm** | Multi-agent coordination (54+ agents) | `npx ruv-swarm mcp start` |
| **flow-nexus** | Cloud orchestration (70+ tools) | `npx flow-nexus@latest mcp start` |
| **github** | Repository management, PRs, issues | `@modelcontextprotocol/server-github` |
| **filesystem** | Local file operations | `@modelcontextprotocol/server-filesystem` |
| **git** | Version control operations | `@modelcontextprotocol/server-git` |
| **python** | Python code execution | `@modelcontextprotocol/server-python` |

## Common Prompts

### SPARC Methodology (Claude Flow)
```
Use SPARC methodology to implement [feature]
Run TDD workflow for [component]
Create a complete test suite for [module]
```

### Multi-Agent Coordination (Ruv-Swarm)
```
Spawn a swarm to refactor [component]
Use hierarchical coordination to build [feature]
Deploy mesh topology for [complex task]
```

### GitHub Integration
```
Create a PR for [changes] with detailed description
Review open issues and prioritize them
Analyze recent commits for [pattern]
```

### Code Analysis
```
Find all TODO comments and create issues
Analyze dependencies and suggest optimizations
Review code quality across the project
```

## Environment Variables

```bash
# Add to ~/.bashrc or ~/.zshrc

# GitHub integration (required for GitHub MCP)
export GITHUB_TOKEN="ghp_your_token_here"

# Claude Flow settings
export CLAUDE_FLOW_AUTO_COMMIT="false"
export CLAUDE_FLOW_AUTO_PUSH="false"
export CLAUDE_FLOW_HOOKS_ENABLED="true"
export CLAUDE_FLOW_TELEMETRY_ENABLED="true"
```

## Accessing MCP in Augment

### Via Settings Panel
1. Open Augment panel in VSCode
2. Click ⚙️ (Settings) in upper right
3. Navigate to "MCP Servers" section
4. View/edit/test configured servers

### Via Easy MCP (One-Click Integrations)
1. Open Augment Settings
2. Go to "Easy MCP" pane
3. Click "+" next to desired integration
4. Paste API token or approve OAuth

## Testing MCP Setup

```bash
# Test Claude Flow
npx claude-flow@alpha --version

# Test Ruv-Swarm
npx ruv-swarm --version

# List available agents
npx claude-flow@alpha agents list
```

## Troubleshooting

### MCP Servers Not Appearing
```bash
# Restart VSCode
# Check Augment extension is latest version
# Verify Node.js: node --version
```

### Permission Errors
```bash
# Fix npm global permissions
mkdir -p ~/.npm-global
npm config set prefix '~/.npm-global'
echo 'export PATH=~/.npm-global/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

### GitHub Token Issues
1. Create token at https://github.com/settings/tokens
2. Required scopes: `repo`, `workflow`, `read:org`
3. Set: `export GITHUB_TOKEN="ghp_..."`

## Integration with Project

### SPARC Workflow
```bash
# Use SPARC commands with MCP
npx claude-flow sparc tdd "voice conversion feature"
npx claude-flow sparc pipeline "complete implementation"
```

### Agent Coordination
```bash
# Initialize swarm
npx claude-flow swarm init --topology mesh

# Spawn agents
npx claude-flow agent spawn --type coder
npx claude-flow agent spawn --type tester
```

## Key Features

### Claude Flow MCP Tools
- `swarm_init` - Initialize coordination topology
- `agent_spawn` - Spawn specialized agents
- `task_orchestrate` - Orchestrate workflows
- `memory_usage` - Track memory and context
- `github_swarm` - GitHub integration
- `benchmark_run` - Performance benchmarking

### Ruv-Swarm Agents (54 Total)
- **Core**: coder, reviewer, tester, planner, researcher
- **Swarm**: hierarchical-coordinator, mesh-coordinator, adaptive-coordinator
- **GitHub**: github-modes, pr-manager, code-review-swarm, issue-tracker
- **SPARC**: sparc-coord, specification, pseudocode, architecture, refinement
- **Specialized**: backend-dev, mobile-dev, ml-developer, cicd-engineer

## Resources

- 📚 [Full Setup Guide](./AUGMENT_MCP_SETUP.md)
- 🔧 [Claude Flow Docs](https://github.com/ruvnet/claude-flow)
- 🌐 [Augment MCP Docs](https://docs.augmentcode.com/setup-augment/mcp)
- 📖 [MCP Specification](https://modelcontextprotocol.io/)
- 📝 [Project CLAUDE.md](../CLAUDE.md)

## Quick Commands

```bash
# Setup
./scripts/setup_augment_mcp.sh

# Test installation
npx claude-flow@alpha --version
npx ruv-swarm --version

# View configuration
cat .vscode/mcp.json
cat .vscode/settings.json

# Check environment
echo $GITHUB_TOKEN
env | grep CLAUDE_FLOW
```

## Support

- Augment Documentation: https://docs.augmentcode.com
- Claude Flow Issues: https://github.com/ruvnet/claude-flow/issues
- AutoVoice Repository: https://github.com/khryptorgraphics/autovoice

