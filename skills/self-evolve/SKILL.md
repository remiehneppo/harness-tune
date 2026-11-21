---
name: self-evolve
description: Protocol for worker agent to submit execution traces and consume promoted harness versions.
---

# Self-Evolution Policy

## When to Submit Traces
- Submit trace on task completion or failure using `trace.submit`.
- Include pass/fail status, step count, and any unrecovered errors.

## When to Refresh Harness
- If the harness registry emits a promoted version matching this agent's model profile, reload the active `HarnessManifest`.
