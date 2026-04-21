#!/usr/bin/env python3
"""
Historical claude-flow swarm orchestrator for AutoVoice.

Launches and coordinates parallel agent swarms for:
- SOTA Dual-Pipeline implementation (P0)
- Training-Inference integration (P1)
- YouTube Artist training pipeline (P1)

Usage:
    python scripts/swarm_orchestrator.py --swarm all
    python scripts/swarm_orchestrator.py --swarm sota-dual-pipeline
    python scripts/swarm_orchestrator.py --swarm youtube-artist --parallel 4
    python scripts/swarm_orchestrator.py --status
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_FILE = PROJECT_ROOT / "config" / "swarm_config.yaml"
AGENT_CONTEXTS_FILE = PROJECT_ROOT / "config" / "agent_contexts.yaml"


@dataclass
class SwarmStatus:
    """Status of a swarm execution."""
    name: str
    phase: int
    total_phases: int
    agents_active: int
    agents_complete: int
    tasks_pending: int
    tasks_complete: int
    started_at: str
    errors: list[str] = field(default_factory=list)


def load_config() -> dict[str, Any]:
    """Load swarm configuration."""
    if not CONFIG_FILE.exists():
        print(f"Error: Config file not found: {CONFIG_FILE}")
        sys.exit(1)

    with open(CONFIG_FILE) as f:
        return yaml.safe_load(f)


def load_agent_contexts() -> dict[str, Any]:
    """Load agent context injection rules."""
    if not AGENT_CONTEXTS_FILE.exists():
        print(f"Warning: Agent contexts file not found: {AGENT_CONTEXTS_FILE}")
        return {}

    with open(AGENT_CONTEXTS_FILE) as f:
        return yaml.safe_load(f)


def run_command(cmd: list[str], capture: bool = False) -> tuple[int, str]:
    """Run a shell command."""
    try:
        if capture:
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode, result.stdout + result.stderr
        else:
            result = subprocess.run(cmd)
            return result.returncode, ""
    except Exception as e:
        return 1, str(e)


def check_claude_flow() -> bool:
    """Check if claude-flow is available."""
    code, output = run_command(["claude-flow", "--version"], capture=True)
    if code != 0:
        print("Error: claude-flow not found. Install with: npm install -g @anthropic/claude-flow")
        return False
    print(f"Found: {output.strip()}")
    return True


def init_queen(config: dict[str, Any]) -> bool:
    """Initialize the Queen coordinator with full project context."""
    print("\n=== Initializing Queen Coordinator ===")

    queen_config = config.get("queen", {})
    context_files = queen_config.get("context_files", [])

    # Build context file list
    context_args = []
    for cf in context_files:
        full_path = PROJECT_ROOT / cf
        if full_path.exists():
            context_args.extend(["--context", str(full_path)])
        else:
            print(f"Warning: Context file not found: {cf}")

    # Initialize hive-mind with hierarchical-mesh topology and neural awareness
    cmd = [
        "claude-flow", "hive-mind", "init",
        "-t", "hierarchical-mesh",
        "--neural",  # Enable neural-aware coordination (98.5% target accuracy)
        "--name", queen_config.get("name", "autovoice-queen"),
    ]

    code, output = run_command(cmd, capture=True)
    if code != 0:
        print(f"Error initializing queen: {output}")
        return False

    print(f"Queen initialized: {queen_config.get('name', 'autovoice-queen')}")
    return True


def build_agent_context(
    agent_name: str,
    agent_config: dict[str, Any],
    swarm_config: dict[str, Any],
    agent_contexts: dict[str, Any],
) -> list[str]:
    """Build the context file list for an agent."""
    context_files = []

    # Add default context
    defaults = agent_contexts.get("defaults", {}).get("required", [])
    for cf in defaults:
        path = PROJECT_ROOT / cf
        if path.exists():
            context_files.append(str(path))

    # Add agent type context
    agent_type = agent_config.get("type", "developer")
    type_context = agent_contexts.get("agent_types", {}).get(agent_type, {}).get("context", [])
    for pattern in type_context:
        # Handle glob patterns
        if "*" in pattern:
            for path in PROJECT_ROOT.glob(pattern):
                if path.is_file():
                    context_files.append(str(path))
        else:
            path = PROJECT_ROOT / pattern
            if path.exists():
                context_files.append(str(path))

    # Add swarm-specific context
    swarm_context = swarm_config.get("context_files", [])
    for pattern in swarm_context:
        if "*" in pattern:
            for path in PROJECT_ROOT.glob(pattern):
                if path.is_file():
                    context_files.append(str(path))
        else:
            path = PROJECT_ROOT / pattern
            if path.exists():
                context_files.append(str(path))

    # Add agent-specific context
    agent_context = agent_config.get("context_files", [])
    for pattern in agent_context:
        if "*" in pattern:
            for path in PROJECT_ROOT.glob(pattern):
                if path.is_file():
                    context_files.append(str(path))
        else:
            path = PROJECT_ROOT / pattern
            if path.exists():
                context_files.append(str(path))

    # Deduplicate while preserving order
    seen = set()
    unique_files = []
    for f in context_files:
        if f not in seen:
            seen.add(f)
            unique_files.append(f)

    return unique_files


# ==============================================================================
# Context Injection - Intelligent content loading with prioritization
# ==============================================================================

# Priority tiers for context files
PRIORITY_TIERS = {
    # Tier 1: Critical - Always include full content
    "critical": [
        "CLAUDE.md",
        "spec.md",
        "plan.md",
        "PROMPT.md",
        "product.md",
        "tech-stack.md",
        "workflow.md",
    ],
    # Tier 2: Important - Include full if space allows
    "important": [
        "test_",
        "_test.py",
        "api.py",
        "requirements.txt",
        "pytest.ini",
    ],
    # Tier 3: Reference - Summarize (signatures + docstrings only)
    "reference": [
        ".py",
        ".ts",
        ".tsx",
    ],
}

# Approximate chars per token for budget estimation
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Estimate token count from text length."""
    return len(text) // CHARS_PER_TOKEN


def get_file_priority(filepath: str) -> int:
    """
    Determine priority tier for a file.
    Returns: 1 (critical), 2 (important), 3 (reference), 4 (skip)
    """
    filename = Path(filepath).name

    # Check critical files
    for pattern in PRIORITY_TIERS["critical"]:
        if pattern in filename:
            return 1

    # Check important files
    for pattern in PRIORITY_TIERS["important"]:
        if pattern in filename:
            return 2

    # Check reference files (source code)
    for pattern in PRIORITY_TIERS["reference"]:
        if filepath.endswith(pattern):
            return 3

    # Everything else - lower priority
    return 4


def extract_python_signatures(content: str, max_lines: int = 100) -> str:
    """
    Extract function/class signatures and docstrings from Python code.
    Produces a condensed summary for reference context.
    """
    import ast
    import textwrap

    try:
        tree = ast.parse(content)
    except SyntaxError:
        # If parsing fails, return first N lines
        lines = content.split("\n")[:max_lines]
        return "\n".join(lines) + "\n# ... (truncated)"

    summary_parts = []

    # Get module docstring
    if ast.get_docstring(tree):
        summary_parts.append(f'"""{ast.get_docstring(tree)}"""')
        summary_parts.append("")

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            # Include imports
            names = ", ".join(alias.name for alias in node.names)
            summary_parts.append(f"import {names}")

        elif isinstance(node, ast.ImportFrom):
            names = ", ".join(alias.name for alias in node.names)
            summary_parts.append(f"from {node.module} import {names}")

        elif isinstance(node, ast.ClassDef):
            # Class definition with docstring
            bases = ", ".join(
                ast.unparse(base) if hasattr(ast, "unparse") else "..."
                for base in node.bases
            )
            class_line = f"class {node.name}({bases}):" if bases else f"class {node.name}:"
            summary_parts.append("")
            summary_parts.append(class_line)

            docstring = ast.get_docstring(node)
            if docstring:
                # Truncate long docstrings
                if len(docstring) > 200:
                    docstring = docstring[:200] + "..."
                summary_parts.append(f'    """{docstring}"""')

            # Include method signatures
            for item in node.body:
                if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                    sig = _format_function_signature(item)
                    summary_parts.append(f"    {sig}")
                    method_doc = ast.get_docstring(item)
                    if method_doc:
                        short_doc = method_doc.split("\n")[0][:100]
                        summary_parts.append(f'        """{short_doc}..."""')
                    summary_parts.append("        ...")

        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            # Top-level function
            summary_parts.append("")
            sig = _format_function_signature(node)
            summary_parts.append(sig)
            docstring = ast.get_docstring(node)
            if docstring:
                short_doc = docstring.split("\n")[0][:100]
                summary_parts.append(f'    """{short_doc}..."""')
            summary_parts.append("    ...")

        elif isinstance(node, ast.Assign):
            # Top-level constants (ALL_CAPS)
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    try:
                        value = ast.unparse(node.value) if hasattr(ast, "unparse") else "..."
                        if len(value) > 50:
                            value = value[:50] + "..."
                        summary_parts.append(f"{target.id} = {value}")
                    except Exception:
                        pass

    result = "\n".join(summary_parts)

    # Ensure we don't exceed max_lines
    lines = result.split("\n")
    if len(lines) > max_lines:
        result = "\n".join(lines[:max_lines]) + "\n# ... (truncated)"

    return result


def _format_function_signature(node: "ast.FunctionDef | ast.AsyncFunctionDef") -> str:
    """Format a function definition signature."""
    import ast

    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"

    # Build argument list
    args = []
    all_args = node.args

    # Positional args
    defaults_offset = len(all_args.args) - len(all_args.defaults)
    for i, arg in enumerate(all_args.args):
        arg_str = arg.arg
        if arg.annotation:
            try:
                arg_str += f": {ast.unparse(arg.annotation)}"
            except Exception:
                pass
        # Add default if exists
        default_idx = i - defaults_offset
        if default_idx >= 0 and default_idx < len(all_args.defaults):
            try:
                default_val = ast.unparse(all_args.defaults[default_idx])
                if len(default_val) > 20:
                    default_val = "..."
                arg_str += f" = {default_val}"
            except Exception:
                pass
        args.append(arg_str)

    # *args
    if all_args.vararg:
        args.append(f"*{all_args.vararg.arg}")

    # **kwargs
    if all_args.kwarg:
        args.append(f"**{all_args.kwarg.arg}")

    args_str = ", ".join(args)

    # Return annotation
    returns = ""
    if node.returns:
        try:
            returns = f" -> {ast.unparse(node.returns)}"
        except Exception:
            pass

    return f"{prefix} {node.name}({args_str}){returns}:"


def summarize_typescript(content: str, max_lines: int = 80) -> str:
    """
    Extract key structures from TypeScript/TSX files.
    Uses regex-based extraction since we don't have a TS parser.
    """
    import re

    summary_parts = []
    lines = content.split("\n")

    # Track what we've extracted
    in_interface = False
    in_function = False
    brace_depth = 0

    for line in lines:
        stripped = line.strip()

        # Imports
        if stripped.startswith("import "):
            summary_parts.append(line)

        # Exports
        elif stripped.startswith("export "):
            if "interface " in stripped or "type " in stripped:
                summary_parts.append(line)
            elif "function " in stripped or "const " in stripped:
                # Just the signature
                summary_parts.append(line.split("{")[0].strip() + " { ... }")
            elif "class " in stripped:
                summary_parts.append(line.split("{")[0].strip() + " { ... }")

        # Interface/type definitions
        elif stripped.startswith("interface ") or stripped.startswith("type "):
            summary_parts.append(line.split("{")[0].strip() + " { ... }")

        # Function declarations
        elif re.match(r"^(async\s+)?function\s+\w+", stripped):
            summary_parts.append(line.split("{")[0].strip() + " { ... }")

        # React components (const Name = () => or function Name())
        elif re.match(r"^const\s+[A-Z]\w+\s*[:=]", stripped):
            summary_parts.append(line.split("=>")[0].strip() + " => { ... }")

    result = "\n".join(summary_parts[:max_lines])
    if len(summary_parts) > max_lines:
        result += "\n// ... (truncated)"

    return result


def inject_context_content(
    context_files: list[str],
    max_tokens: int = 50000,
) -> str:
    """
    Read context files and build a condensed prompt for the agent.

    Prioritization strategy:
    - Tier 1 (Critical): CLAUDE.md, spec.md, plan.md - always full content
    - Tier 2 (Important): Test files, API files - full if space allows
    - Tier 3 (Reference): Source code - summarized (signatures + docstrings)

    Args:
        context_files: List of absolute file paths to include
        max_tokens: Maximum token budget (default 50k)

    Returns:
        Condensed context string ready for agent injection
    """
    if not context_files:
        return ""

    # Sort files by priority
    prioritized: dict[int, list[tuple[str, Path]]] = {1: [], 2: [], 3: [], 4: []}

    for filepath in context_files:
        path = Path(filepath)
        if not path.exists():
            continue
        priority = get_file_priority(filepath)
        prioritized[priority].append((filepath, path))

    # Build context with budget tracking
    context_parts: list[str] = []
    tokens_used = 0
    files_included = 0
    files_summarized = 0

    def add_content(label: str, content: str, is_summary: bool = False) -> bool:
        """Add content if within budget. Returns True if added."""
        nonlocal tokens_used, files_included, files_summarized

        content_tokens = estimate_tokens(content)
        if tokens_used + content_tokens > max_tokens:
            return False

        marker = " [SUMMARIZED]" if is_summary else ""
        context_parts.append(f"\n{'='*60}\n# {label}{marker}\n{'='*60}\n")
        context_parts.append(content)
        tokens_used += content_tokens
        files_included += 1
        if is_summary:
            files_summarized += 1
        return True

    # Process Tier 1: Critical files (always full)
    for filepath, path in prioritized[1]:
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            label = path.name
            if not add_content(label, content):
                # Even critical files can't exceed budget - truncate
                max_chars = (max_tokens - tokens_used) * CHARS_PER_TOKEN
                if max_chars > 1000:
                    truncated = content[:max_chars] + "\n\n... [TRUNCATED - budget exceeded]"
                    add_content(label, truncated)
                break
        except Exception as e:
            context_parts.append(f"\n# Error reading {path.name}: {e}\n")

    # Process Tier 2: Important files (full if space)
    for filepath, path in prioritized[2]:
        if tokens_used >= max_tokens * 0.8:  # Reserve 20% for reference
            break
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            label = f"{path.parent.name}/{path.name}"
            add_content(label, content)
        except Exception:
            pass

    # Process Tier 3: Reference files (summarized)
    for filepath, path in prioritized[3]:
        if tokens_used >= max_tokens * 0.95:  # Leave 5% buffer
            break
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            label = f"{path.parent.name}/{path.name}"

            # Summarize based on file type
            if filepath.endswith(".py"):
                summary = extract_python_signatures(content, max_lines=60)
            elif filepath.endswith((".ts", ".tsx")):
                summary = summarize_typescript(content, max_lines=50)
            else:
                # For other files, just take first N lines
                lines = content.split("\n")[:30]
                summary = "\n".join(lines) + "\n# ... (truncated)"

            add_content(label, summary, is_summary=True)
        except Exception:
            pass

    # Add metadata header
    header = f"""# Agent Context Injection
# Files: {files_included} ({files_summarized} summarized)
# Tokens: ~{tokens_used:,} / {max_tokens:,} budget
# Priority breakdown: {len(prioritized[1])} critical, {len(prioritized[2])} important, {len(prioritized[3])} reference
"""

    return header + "".join(context_parts)


def spawn_agent(
    agent_name: str,
    agent_config: dict[str, Any],
    swarm_name: str,
    swarm_config: dict[str, Any],
    agent_contexts: dict[str, Any],
    use_claude: bool = True,
    inject_full_context: bool = True,
    context_token_budget: int = 50000,
) -> bool:
    """
    Spawn a single agent into the hive with injected context.

    Args:
        agent_name: Name of the agent to spawn
        agent_config: Agent-specific configuration
        swarm_name: Name of the parent swarm
        swarm_config: Swarm configuration
        agent_contexts: Context injection rules
        use_claude: Whether to launch Claude Code
        inject_full_context: Whether to inject full context content (vs just file list)
        context_token_budget: Token budget for context injection
    """
    print(f"\n  Spawning agent: {agent_name}")

    # Build context file list
    context_files = build_agent_context(agent_name, agent_config, swarm_config, agent_contexts)
    print(f"    Context files: {len(context_files)}")

    # Build the task description
    responsibility = agent_config.get("responsibility", "Execute assigned tasks")
    phase = agent_config.get("phase", 1)
    outputs = agent_config.get("outputs", [])
    agent_type = agent_config.get("type", "developer")
    depends_on = agent_config.get("depends_on", [])
    gpu_required = agent_config.get("gpu_required", False)

    # Header with agent metadata
    task_header = f"""
# Agent Assignment
================================================================================
Swarm: {swarm_name}
Agent: {agent_name}
Type: {agent_type}
Phase: {phase}
Track: {swarm_config.get('track', 'N/A')}
GPU Required: {gpu_required}
Dependencies: {depends_on if depends_on else 'None'}

## Responsibility
{responsibility}

## Expected Outputs
{chr(10).join(f'- {o}' for o in outputs)}

## Workflow Rules
1. Follow TDD: Write tests FIRST, then implement
2. Report progress: Update beads tasks (`bd update <id> --status in_progress`)
3. Share discoveries: Write to cipher memory for cross-agent learning
4. No fallback behavior: Raise errors, never pass silently
5. Atomic commits: One feature per commit, run tests before committing

================================================================================
"""

    # Inject full context content if enabled
    if inject_full_context and context_files:
        print(f"    Injecting context (~{context_token_budget:,} token budget)...")
        context_content = inject_context_content(context_files, max_tokens=context_token_budget)
        context_section = f"""
# Injected Context
{context_content}

================================================================================
"""
    else:
        # Just list the files
        context_section = f"""
# Context Files (read these as needed)
{chr(10).join(f'- {f}' for f in context_files[:20])}
{'...' if len(context_files) > 20 else ''}

================================================================================
"""

    task_desc = task_header + context_section

    # Always write context to file to avoid command line issues
    context_dir = PROJECT_ROOT / ".swarm_contexts"
    context_dir.mkdir(exist_ok=True)
    context_file_path = context_dir / f"{swarm_name}-{agent_name}.md"
    context_file_path.write_text(task_desc, encoding="utf-8")
    print(f"    Context written to: {context_file_path}")

    # Short prompt referencing the context file
    short_prompt = f"""[{swarm_name}] Agent: {agent_name}
Phase: {phase} | Type: {agent_type}
Responsibility: {responsibility}

IMPORTANT: Read your full context and instructions from:
{context_file_path}

Start by reading this file, then execute your assigned tasks."""

    # Spawn via claude-flow hive-mind with neural awareness
    cmd = [
        "claude-flow", "hive-mind", "spawn",
        "--neural",  # Enable neural-aware task prediction
        "-n", "1",
        "--name", f"{swarm_name}-{agent_name}",
    ]

    if use_claude:
        cmd.append("--claude")
        cmd.extend(["-o", short_prompt.strip()])
        # Don't capture output for Claude - it's interactive and shouldn't block
        code, output = run_command(cmd, capture=False)
    else:
        code, output = run_command(cmd, capture=True)

    if code != 0:
        print(f"    Error spawning {agent_name}: {output}")
        return False

    print(f"    Agent {agent_name} spawned (phase {phase}, context: {context_file_path.name})")
    return True


def execute_swarm_phase(
    swarm_name: str,
    swarm_config: dict[str, Any],
    phase: int,
    agent_contexts: dict[str, Any],
    parallel: int = 10,
) -> bool:
    """Execute all agents in a specific phase."""
    agents = swarm_config.get("agents", {})

    # Filter agents for this phase
    phase_agents = {
        name: cfg for name, cfg in agents.items()
        if cfg.get("phase", 1) == phase
    }

    if not phase_agents:
        print(f"  No agents in phase {phase}")
        return True

    print(f"\n=== {swarm_name} Phase {phase}: {len(phase_agents)} agents ===")

    # Check dependencies
    ready_agents = []
    blocked_agents = []

    for name, cfg in phase_agents.items():
        deps = cfg.get("depends_on", [])
        if deps:
            # Check if all dependencies are complete
            # In production, this would check beads/cipher for completion status
            blocked_agents.append((name, deps))
        else:
            ready_agents.append((name, cfg))

    # Spawn parallel agents first
    parallel_agents = [(n, c) for n, c in ready_agents if c.get("parallel", False)]
    sequential_agents = [(n, c) for n, c in ready_agents if not c.get("parallel", False)]

    # Spawn parallel agents (up to limit)
    for i in range(0, len(parallel_agents), parallel):
        batch = parallel_agents[i:i + parallel]
        print(f"\n  Spawning parallel batch: {[n for n, _ in batch]}")
        for name, cfg in batch:
            spawn_agent(name, cfg, swarm_name, swarm_config, agent_contexts)

    # Spawn sequential agents one by one
    for name, cfg in sequential_agents:
        spawn_agent(name, cfg, swarm_name, swarm_config, agent_contexts)

    # Report blocked agents
    if blocked_agents:
        print(f"\n  Blocked agents (waiting on dependencies):")
        for name, deps in blocked_agents:
            print(f"    - {name} -> waiting for: {deps}")

    return True


def execute_swarm(
    swarm_name: str,
    config: dict[str, Any],
    agent_contexts: dict[str, Any],
    parallel: int = 10,
    start_phase: int = 1,
) -> bool:
    """Execute a complete swarm."""
    swarms = config.get("swarms", {})

    if swarm_name not in swarms:
        print(f"Error: Unknown swarm: {swarm_name}")
        print(f"Available swarms: {list(swarms.keys())}")
        return False

    swarm_config = swarms[swarm_name]
    priority = swarm_config.get("priority", "P2")
    track = swarm_config.get("track", "N/A")
    dependencies = swarm_config.get("dependencies", [])

    print(f"\n{'='*60}")
    print(f"SWARM: {swarm_name}")
    print(f"Priority: {priority}")
    print(f"Track: {track}")
    print(f"{'='*60}")

    # Check swarm dependencies
    if dependencies:
        print(f"\nDependencies: {dependencies}")
        # In production, check if dependent swarms have completed their phases
        # For now, just warn
        print("  Warning: Dependency checking not yet implemented")

    # Get all phases
    agents = swarm_config.get("agents", {})
    phases = sorted(set(cfg.get("phase", 1) for cfg in agents.values()))

    print(f"\nPhases: {phases}")
    print(f"Total agents: {len(agents)}")

    # Execute each phase
    for phase in phases:
        if phase < start_phase:
            print(f"\n  Skipping phase {phase} (start_phase={start_phase})")
            continue

        if not execute_swarm_phase(swarm_name, swarm_config, phase, agent_contexts, parallel):
            print(f"\n  Phase {phase} failed!")
            return False

    return True


def show_status(config: dict[str, Any]) -> None:
    """Show status of all swarms."""
    print("\n=== Swarm Orchestration Status ===\n")

    # Check hive-mind status
    code, output = run_command(["claude-flow", "hive-mind", "status"], capture=True)
    if code == 0:
        print("Hive Mind Status:")
        print(output)
    else:
        print("Hive Mind: Not initialized")

    # Show swarm configurations
    swarms = config.get("swarms", {})
    print(f"\nConfigured Swarms: {len(swarms)}")

    for name, cfg in swarms.items():
        priority = cfg.get("priority", "P2")
        track = cfg.get("track", "N/A")
        agents = cfg.get("agents", {})
        deps = cfg.get("dependencies", [])

        print(f"\n  {name}:")
        print(f"    Priority: {priority}")
        print(f"    Track: {track}")
        print(f"    Agents: {len(agents)}")
        print(f"    Dependencies: {deps if deps else 'None'}")

        # Show phases
        phases = {}
        for agent_name, agent_cfg in agents.items():
            phase = agent_cfg.get("phase", 1)
            if phase not in phases:
                phases[phase] = []
            phases[phase].append(agent_name)

        print(f"    Phases:")
        for phase, phase_agents in sorted(phases.items()):
            print(f"      Phase {phase}: {len(phase_agents)} agents")


def shutdown_swarms() -> None:
    """Shutdown all active swarms."""
    print("\n=== Shutting down swarms ===")

    code, output = run_command(["claude-flow", "hive-mind", "shutdown"], capture=True)
    if code == 0:
        print("Swarms shutdown successfully")
    else:
        print(f"Shutdown warning: {output}")


def main():
    parser = argparse.ArgumentParser(
        description="Claude-Flow Smart Swarm Orchestrator for AutoVoice"
    )
    parser.add_argument(
        "--swarm",
        choices=["all", "sota-dual-pipeline", "training-inference", "youtube-artist"],
        help="Which swarm(s) to execute",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=10,
        help="Max parallel agents (default: 10)",
    )
    parser.add_argument(
        "--phase",
        type=int,
        default=1,
        help="Start from phase N (default: 1)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show swarm status",
    )
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="Only initialize queen, don't spawn agents",
    )
    parser.add_argument(
        "--shutdown",
        action="store_true",
        help="Shutdown all active swarms",
    )
    parser.add_argument(
        "--no-claude",
        action="store_true",
        help="Don't launch Claude Code for agents (dry run)",
    )

    args = parser.parse_args()

    # Load configurations
    config = load_config()
    agent_contexts = load_agent_contexts()

    # Handle status
    if args.status:
        show_status(config)
        return

    # Handle shutdown
    if args.shutdown:
        shutdown_swarms()
        return

    # Check claude-flow
    if not check_claude_flow():
        sys.exit(1)

    # Initialize queen
    if not init_queen(config):
        sys.exit(1)

    if args.init_only:
        print("\nQueen initialized. Use --swarm to spawn agents.")
        return

    # Execute swarms
    if args.swarm:
        swarms_to_run = []

        if args.swarm == "all":
            # Run in priority order
            swarms_to_run = ["sota-dual-pipeline", "youtube-artist", "training-inference"]
        else:
            swarms_to_run = [args.swarm]

        for swarm_name in swarms_to_run:
            success = execute_swarm(
                swarm_name,
                config,
                agent_contexts,
                parallel=args.parallel,
                start_phase=args.phase,
            )

            if not success:
                print(f"\nSwarm {swarm_name} execution failed!")
                sys.exit(1)

        print(f"\n{'='*60}")
        print("All swarms launched successfully!")
        print("Monitor progress with: python scripts/swarm_orchestrator.py --status")
        print("Shutdown with: python scripts/swarm_orchestrator.py --shutdown")
        print(f"{'='*60}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
