# Contributing to Omnigent

Thank you for your interest in contributing! Omnigent is an open framework and we welcome contributions of all kinds.

## Ways to Contribute

- **Bug reports** — Open an issue with a minimal reproduction
- **Feature requests** — Describe the use case, not just the solution
- **Domain implementations** — Share your agent built on Omnigent (security, DevOps, code quality, etc.)
- **Documentation** — Fix typos, clarify concepts, add examples
- **Code** — Fix bugs, improve performance, add tests

## Development Setup

```bash
# Clone the repo
git clone https://github.com/francescostabile/omnigent.git
cd omnigent

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run type checking
mypy src/omnigent/

# Run linter
ruff check src/
```

## Code Style

- **Python 3.11+** with type hints everywhere
- **Ruff** for linting (config in pyproject.toml)
- **No domain-specific code** in the core — use registries and subclasses
- Docstrings on all public classes and functions
- Tests for all new functionality

## Pull Request Process

1. Fork the repo and create a feature branch from `main`
2. Write tests for your changes
3. Ensure all 109+ tests pass: `pytest`
4. Ensure no type errors: `mypy src/omnigent/`
5. Ensure linting passes: `ruff check src/`
6. Update documentation if needed
7. Submit a PR with a clear description of what and why

## Architecture Guidelines

Omnigent follows strict principles. Before contributing, please read [ARCHITECTURE.md](ARCHITECTURE.md).

**Key rules:**
- The core must remain **domain-agnostic** — no security, code quality, or other domain logic
- All domain-specific behavior goes through **registries** (dicts that start empty)
- Complex domain logic uses **subclass hooks** (ReasoningGraph, DomainProfile, Agent)
- Every component must be **independently testable** without LLM calls
- **Never crash** — extractors, reflectors, and plugins must catch all exceptions

## Adding a New Registry

If you need a new extension point:

1. Create the registry dict in the appropriate module (e.g., `MY_REGISTRY: dict = {}`)
2. Create a core API function that reads from it
3. Add tests that verify it works when empty AND when populated
4. Document the expected signature in the module docstring
5. Add an example in `examples/codelens/registries.py`

## Reporting Security Issues

If you find a security vulnerability, please **do not** open a public issue. Email the maintainer directly.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
