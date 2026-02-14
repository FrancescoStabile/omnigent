"""
Omnigent — Entry Point

Usage:
    omnigent                      # Interactive mode
    omnigent --mcp                # MCP server (stdio)
    omnigent --mcp-http           # MCP server (HTTP)
    omnigent --verbose            # Debug logging
"""

import argparse
import sys

from omnigent import __version__


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="omnigent",
        description="Universal Autonomous Agent Framework",
    )
    parser.add_argument("--version", action="version", version=f"omnigent {__version__}")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--mcp", action="store_true", help="Run as MCP server (stdio)")
    parser.add_argument("--mcp-http", action="store_true", help="Run as MCP server (HTTP)")

    args = parser.parse_args()

    if args.verbose:
        from omnigent.logging_config import setup_logging
        setup_logging(verbose=True)

    if args.mcp or args.mcp_http:
        print("[Omnigent] MCP mode — implement your domain MCP server using omnigent.mcp_server", file=sys.stderr)
        sys.exit(0)

    print(f"Omnigent v{__version__} — Universal Agent Framework")
    print("This is the core framework. Build your domain agent on top of it.")
    print("See README.md and examples/ for how to get started.")


if __name__ == "__main__":
    main()
