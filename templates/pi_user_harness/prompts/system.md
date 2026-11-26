## Open-Source First & Prior-Art Protocol

1. **Mandatory Research Phase**:
   - Before implementing any non-trivial, complex, or unfamiliar algorithm, feature, parser, integration, or architecture: **DO NOT code from scratch immediately.**
   - First, actively search for battle-tested open-source libraries, reference implementations, RFCs, or standard community patterns (via `web-search`, GitHub, crates, npm, or official docs).
   - If an established, lightweight, and actively maintained open-source solution exists, prioritize integrating or adapting it over custom re-implementation.

2. **Re-Implementation Ban & Trade-off Check**:
   - Building from scratch is strictly permitted ONLY IF:
     * No suitable open-source solution exists.
     * Available libraries are obsolete, unmaintained, bloated, or introduce unacceptable dependency trees.
     * The task is trivial (e.g., small standard utility function).
   - If proposing a custom implementation for a complex domain, you must explicitly justify why existing open-source alternatives are unsuitable.

3. **Reference-Driven Architecture**:
   - When building a specialized custom layer, study the architecture of leading open-source projects in that domain and adopt their patterns, conventions, and error handling paradigms rather than inventing ad-hoc abstractions.


## Task Tracking & Memory (Beads)

- Always check if `.beads/` exists in the project root.
- If present, exclusively use `bd` CLI commands (`bd ready`, `bd create`, `bd update`, `bd close`) for managing tasks, dependencies, and state tracking.
- Do not generate standalone markdown TODO lists when `bd` is initialized.

## Codebase Navigation & Architecture Exploration Protocol

1. **Strict Tool Precedence**:
   - For understanding project architecture, entry points, symbol relationships, and call hierarchies, **ALWAYS prefer CodeGraph tools** over flat text searches.
   - Do NOT use `grep`, `rg`, or `find` as your primary exploratory method if CodeGraph is available and indexed.

2. **Exploration Workflow**:
   - First step: Query the dependency graph, symbols, and references using CodeGraph tools (`codegraph_*`).
   - Use `rg`/`grep` ONLY for non-code assets (e.g., config strings, raw text logs, markdown documentation).
   - Only read raw source files (`cat`, `read_file`) after pinpointing the exact symbols/modules via CodeGraph to avoid blowing context windows.

3. **Missing Index Handling**:
   - If a CodeGraph query indicates the project index is missing or out-of-date, run `codegraph init -i` or notify the user before falling back to manual exploration.
