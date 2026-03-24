# CLAUDE.md

Project conventions and guidelines for the `llm_swarm` codebase.

## Language

- All code, comments, docstrings, commit messages, and documentation must be written in **English**.
- All comments must end with a **period** (`.`).

## Project Structure

```
llm_swarm/
├── src/                    # All source code lives here.
│   ├── sim/                # Physics simulation core.
│   ├── envs/               # PettingZoo MARL environment wrappers.
│   ├── llm/                # LLM integration layer.
│   ├── agents/             # MARL agent implementations.
│   ├── eval/               # ZSC evaluation framework.
│   └── main.py             # Pygame visualization entry point.
├── scripts/                # Training and evaluation entry scripts.
├── configs/                # YAML configuration files.
├── tests/                  # Unit and integration tests.
├── pyproject.toml
└── uv.lock
```

## Dependency Management

- Use **uv** for all dependency management.
- `uv add <pkg>` to add a runtime dependency.
- `uv add --group dev <pkg>` to add a dev dependency.
- `uv sync` to install all dependencies.
- `uv run <cmd>` to run commands within the virtual environment.
- Never use `pip install` directly.

## Configuration as Single Source of Truth

- All runtime parameters (map layout, obstacle list, robot count, cargo shape, seed,
  reward weights, LLM trigger thresholds, etc.) must be read from **YAML configs**
  or **CLI arguments** — never hardcoded in source.
- Source code may define **default values** in dataclass fields or function signatures,
  but these must always be overridable via config.
- No magic constants for maps, obstacles, or seeds scattered across modules.
  Centralise them in `configs/` YAML files.

## LLM I/O Convention

- LLM output **must** be valid JSON conforming to a predefined schema.
- Use `command_parser.py` with **JSON Schema validation** (e.g., `jsonschema` or
  Pydantic) to verify every LLM response before acting on it.
- Free-form text from the LLM is allowed **only** in the `reasoning` field and must
  **never** drive control logic or be parsed for commands.
- If JSON parsing or schema validation fails, fall back to the **default command**
  (all robots normal force, no route bias) — never crash or block.

## Async LLM Contract

- LLM calls **must** run in a background thread or process. The simulation main loop
  must **never** block on an LLM response.
- The main loop reads the commander's **latest cached command**. If no new command has
  arrived, the previous command remains in effect.
- Each cached command has a **TTL** (time-to-live, default ~5 seconds / 300 steps).
  Once expired, the system reverts to the default command to prevent stale guidance.

## Code Style

- Follow **PEP 8** strictly.
- Maximum line length: **88 characters** (Black default).
- Use **type hints** for all function signatures and class attributes.
- Use `from __future__ import annotations` at the top of every module.

### Naming Conventions

- Classes: `PascalCase` (e.g., `TransportObject`, `AsyncLLMBridge`).
- Functions and variables: `snake_case` (e.g., `compute_reward`, `force_weights`).
- Constants: `UPPER_SNAKE_CASE` (e.g., `FORCE_MAX`, `ROBOT_COLORS`).
- Private members: single leading underscore (e.g., `_attach_idx`).
- Module-level dunder names: only `__all__` when needed.

### Docstrings

- Use **Google-style** docstrings for all public classes and functions.
- Include `Args:`, `Returns:`, and `Raises:` sections where applicable.
- All docstrings must end with a period.

```python
def compute_reward(prev_dist: float, curr_dist: float) -> float:
    """Compute the distance-based reward for a single step.

    Args:
        prev_dist: Distance to goal at the previous step.
        curr_dist: Distance to goal at the current step.

    Returns:
        The scalar reward value.
    """
```

### Imports

- Group imports in order: stdlib, third-party, local.
- Use absolute imports (e.g., `from src.sim.world import World`).
- One import per line for `from` imports when there are more than 3 names.

## Linting and Formatting

Run all checks before committing:

```bash
# Format code.
uv run black .

# Lint code.
uv run ruff check .

# Type check.
uv run mypy . --strict
```

### Tool Configuration (in pyproject.toml)

```toml
[tool.black]
line-length = 88
target-version = ["py310"]

[tool.ruff]
line-length = 88
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "UP", "B", "SIM"]

[tool.mypy]
python_version = "3.10"
```

## Testing

- Use **pytest** for all tests.
- Test files go in `tests/` and mirror the `src/` structure.
- Run tests with `uv run pytest tests/`.

## Git Conventions

- Write commit messages in English, imperative mood (e.g., "Add reward shaper module.").
- Keep commits focused on a single logical change.
