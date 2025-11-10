# Augment MCP Configuration Setup Guide

This guide explains how to configure Model Context Protocol (MCP) servers with Augment in VSCode for the AutoVoice project.

## Overview

MCP (Model Context Protocol) allows Augment Agent to access external tools and data sources, expanding its capabilities beyond code editing. This setup integrates:

- **Claude Flow**: SPARC methodology and agent orchestration
- **Ruv-Swarm**: Multi-agent coordination with 54+ specialized agents
- **Flow-Nexus**: Cloud-based orchestration (optional, requires auth)
- **GitHub**: Repository management and PR workflows
- **Filesystem**: Local file operations
- **Git**: Version control operations
- **Python**: Code execution and analysis

## Quick Start

### 1. Install Required Dependencies

```bash
# Install Claude Flow (required)
npm install -g claude-flow@alpha

# Install Ruv-Swarm (optional but recommended)
npm install -g ruv-swarm

# Install Flow-Nexus (optional, for cloud features)
npm install -g flow-nexus@latest
```

### 2. Configuration Files

The MCP configuration has been added to:

- **`.vscode/settings.json`** - VSCode-specific Augment settings
- **`.vscode/mcp.json`** - MCP server configurations

### 3. Verify Installation

Open VSCode and check the Augment extension:

1. Open the Augment panel in VSCode
2. Click the settings icon (⚙️) in the upper right
3. Navigate to the "MCP Servers" section
4. You should see all configured servers listed

## Available MCP Servers

### Core Servers

#### Claude Flow
- **Purpose**: SPARC methodology, TDD workflows, agent orchestration
- **Command**: `npx claude-flow@alpha mcp start`
- **Features**:
  - Swarm initialization and coordination
  - Agent spawning and task orchestration
  - Memory management and neural training
  - GitHub integration
  - Performance benchmarking

#### Ruv-Swarm
- **Purpose**: Enhanced multi-agent coordination
- **Command**: `npx ruv-swarm mcp start`
- **Features**:
  - 54+ specialized agents (coder, reviewer, tester, etc.)
  - Hierarchical, mesh, and adaptive coordination
  - Byzantine fault tolerance
  - Consensus building and CRDT synchronization

### Integration Servers

#### GitHub
- **Purpose**: Repository management, PRs, issues
- **Setup**: Requires `GITHUB_TOKEN` environment variable
- **Usage**: "Create a PR for this feature" or "Review open issues"

#### Filesystem
- **Purpose**: Local file operations
- **Scope**: `/home/kp/autovoice` directory
- **Usage**: "Read all Python files in src/" or "Find config files"

#### Git
- **Purpose**: Version control operations
- **Scope**: AutoVoice repository
- **Usage**: "Show recent commits" or "Create a new branch"

#### Python
- **Purpose**: Python code execution and analysis
- **Usage**: "Run this Python script" or "Analyze dependencies"

## Using MCP with Augment Agent

### Example Prompts

**With Claude Flow:**
```
Use SPARC methodology to implement a new voice conversion feature
```

**With Ruv-Swarm:**
```
Spawn a swarm of agents to refactor the audio processing pipeline
```

**With GitHub:**
```
Create a PR for the CUDA optimization changes with a detailed description
```

**With Filesystem + Git:**
```
Find all TODO comments in the codebase and create issues for them
```

## Advanced Configuration

### Environment Variables

Set these in your shell profile (`~/.bashrc` or `~/.zshrc`):

```bash
# GitHub integration
export GITHUB_TOKEN="your_github_personal_access_token"

# Claude Flow settings
export CLAUDE_FLOW_AUTO_COMMIT="false"
export CLAUDE_FLOW_AUTO_PUSH="false"
export CLAUDE_FLOW_HOOKS_ENABLED="true"
```

### Custom MCP Servers

To add custom MCP servers, edit `.vscode/mcp.json`:

```json
{
  "mcpServers": {
    "your-custom-server": {
      "command": "npx",
      "args": ["-y", "your-mcp-package"],
      "env": {
        "API_KEY": "${YOUR_API_KEY}"
      },
      "description": "Your custom MCP server"
    }
  }
}
```

## Troubleshooting

### MCP Servers Not Showing Up

1. Restart VSCode
2. Check the Augment extension is updated to the latest version
3. Verify Node.js and npm are installed: `node --version && npm --version`

### Permission Errors

```bash
# Fix npm global permissions
mkdir -p ~/.npm-global
npm config set prefix '~/.npm-global'
echo 'export PATH=~/.npm-global/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

### GitHub Token Issues

1. Create a Personal Access Token at https://github.com/settings/tokens
2. Required scopes: `repo`, `workflow`, `read:org`
3. Set in environment: `export GITHUB_TOKEN="ghp_..."`

## Integration with Existing Setup

This MCP configuration integrates with your existing `.claude/` setup:

- **`.claude/settings.json`** - Claude Code hooks and permissions
- **`.claude/agents/`** - 54 available agent definitions
- **`CLAUDE.md`** - SPARC methodology and workflow guidelines

The MCP servers provide the **execution layer** while the `.claude/` configuration provides the **coordination layer**.

## Next Steps

1. ✅ MCP configuration files created
2. 📦 Install required npm packages
3. 🔑 Set up environment variables (especially `GITHUB_TOKEN`)
4. 🧪 Test with a simple prompt: "List all MCP servers available"
5. 🚀 Start using Agent with MCP-enhanced capabilities

## Resources

- [Augment MCP Documentation](https://docs.augmentcode.com/setup-augment/mcp)
- [Claude Flow GitHub](https://github.com/ruvnet/claude-flow)
- [MCP Specification](https://modelcontextprotocol.io/)
- [AutoVoice CLAUDE.md](../CLAUDE.md) - Project-specific guidelines

## Support

For issues or questions:
- Check the Augment documentation
- Review the Claude Flow README
- Open an issue in the AutoVoice repository

