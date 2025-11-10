#!/bin/bash
# Setup Augment MCP Configuration for AutoVoice
# This script installs required MCP servers and configures Augment in VSCode

set -e

echo "🚀 Setting up Augment MCP Configuration for AutoVoice"
echo "=================================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check prerequisites
echo -e "\n${YELLOW}Checking prerequisites...${NC}"

if ! command -v node &> /dev/null; then
    echo -e "${RED}❌ Node.js is not installed. Please install Node.js first.${NC}"
    exit 1
fi

if ! command -v npm &> /dev/null; then
    echo -e "${RED}❌ npm is not installed. Please install npm first.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Node.js $(node --version) found${NC}"
echo -e "${GREEN}✅ npm $(npm --version) found${NC}"

# Install MCP servers
echo -e "\n${YELLOW}Installing MCP servers...${NC}"

echo "📦 Installing Claude Flow (required)..."
npm install -g claude-flow@alpha || {
    echo -e "${RED}❌ Failed to install Claude Flow${NC}"
    exit 1
}
echo -e "${GREEN}✅ Claude Flow installed${NC}"

echo "📦 Installing Ruv-Swarm (recommended)..."
npm install -g ruv-swarm || {
    echo -e "${YELLOW}⚠️  Ruv-Swarm installation failed (optional)${NC}"
}

echo "📦 Installing Flow-Nexus (optional)..."
npm install -g flow-nexus@latest || {
    echo -e "${YELLOW}⚠️  Flow-Nexus installation failed (optional)${NC}"
}

# Install standard MCP servers
echo -e "\n${YELLOW}Installing standard MCP servers...${NC}"

MCP_SERVERS=(
    "@modelcontextprotocol/server-github"
    "@modelcontextprotocol/server-filesystem"
    "@modelcontextprotocol/server-git"
    "@modelcontextprotocol/server-python"
)

for server in "${MCP_SERVERS[@]}"; do
    echo "📦 Installing $server..."
    npm install -g "$server" || {
        echo -e "${YELLOW}⚠️  $server installation failed (optional)${NC}"
    }
done

# Check for GitHub token
echo -e "\n${YELLOW}Checking GitHub configuration...${NC}"

if [ -z "$GITHUB_TOKEN" ]; then
    echo -e "${YELLOW}⚠️  GITHUB_TOKEN not set${NC}"
    echo "To enable GitHub integration, set your token:"
    echo "  export GITHUB_TOKEN='your_github_token'"
    echo "Get a token at: https://github.com/settings/tokens"
else
    echo -e "${GREEN}✅ GITHUB_TOKEN is set${NC}"
fi

# Verify configuration files
echo -e "\n${YELLOW}Verifying configuration files...${NC}"

if [ -f ".vscode/mcp.json" ]; then
    echo -e "${GREEN}✅ .vscode/mcp.json exists${NC}"
else
    echo -e "${RED}❌ .vscode/mcp.json not found${NC}"
fi

if [ -f ".vscode/settings.json" ]; then
    echo -e "${GREEN}✅ .vscode/settings.json exists${NC}"
else
    echo -e "${RED}❌ .vscode/settings.json not found${NC}"
fi

# Test Claude Flow
echo -e "\n${YELLOW}Testing Claude Flow installation...${NC}"
if npx claude-flow@alpha --version &> /dev/null; then
    echo -e "${GREEN}✅ Claude Flow is working${NC}"
else
    echo -e "${RED}❌ Claude Flow test failed${NC}"
fi

# Summary
echo -e "\n${GREEN}=================================================="
echo "✅ Augment MCP Setup Complete!"
echo "==================================================${NC}"

echo -e "\n${YELLOW}Next Steps:${NC}"
echo "1. Restart VSCode to load the new MCP configuration"
echo "2. Open the Augment panel and check Settings → MCP Servers"
echo "3. Set GITHUB_TOKEN if not already set:"
echo "   export GITHUB_TOKEN='your_token'"
echo "4. Test with a prompt: 'List all available MCP servers'"
echo ""
echo "📚 Documentation: docs/AUGMENT_MCP_SETUP.md"
echo "🔧 Configuration: .vscode/mcp.json"
echo ""
echo -e "${GREEN}Happy coding with Augment! 🚀${NC}"

