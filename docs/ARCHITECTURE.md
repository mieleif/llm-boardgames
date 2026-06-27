# Architecture

The repository is a **game engine + LLM benchmarking framework**. The aim is to
compare LLMs (OpenAI, Gemini, Anthropic) on a board game across four axes: OCR /
board reading, tool use, rule understanding, and play skill — and to grow into a
multi-game benchmark.

```
toy_battle/
  engine/          pure-Python rules engine (no third-party deps)
    troops.py        troop catalog (force + effect key)
    board.py         Board graph model + JSON loader (data-driven terrains)
    state.py         GameState / PlayerState / Tile, setup, hidden info, clone()
    connectivity.py  HQ-connection (BFS over occupied bases) + region control
    moves.py         Move / Decision types, legal-move generation
    effects.py       troop + special-base effect handlers (registry)
    rules.py         apply_move(), effect ordering, region scoring, win/end
  boards/
    plaine_des_chateaux.json   the starter terrain (nodes/edges/regions)
  render/
    text.py          per-player observation dict + text rendering
    image.py         programmatic PNG render (vision/OCR + human CLI)
  agents/
    base.py          Agent interface + telemetry (AgentStats)
    random_agent.py, heuristic_agent.py, human_cli.py
    llm/
      base_llm.py    provider-agnostic pipeline (prompt, tool call, retry,
                     fallback, token/cost/latency)
      prompts.py     rules briefing (paraphrased) + system prompt
      anthropic_agent.py / openai_agent.py / gemini_agent.py
  match/
    runner.py        play_game(): one game, JSONL turn log
    benchmark.py     round-robin, aggregation, Elo, CSV/JSON summary
  cli.py             play / benchmark / boards / render
tests/               engine, board, effects, LLM-pipeline (no network)
```

## Key design points

- **Single source of truth for legality.** `legal_moves(state)` enumerates every
  legal top-level move; `apply_move` validates membership. Agents (including
  LLMs) always pick from this list, so they can't desync from the rules — an
  illegal *attempt* is recorded as a metric rather than corrupting state.

- **Effect sub-decisions via a callback.** Effects that need a choice (Cap'taine
  extra placement, Mastok target, plaine return) raise a `Decision` whose options
  are pre-validated; the engine asks the acting agent's `decide()` to pick one.
  This keeps the top-level action space small while still routing every choice
  through the agent (good for testing LLM tool use).

- **Deterministic & replayable.** All randomness flows through a seeded RNG on
  the state; `state.clone()` enables 1-ply search (used by the heuristic bot) and
  reproducible matches. The JSONL log replays a game move by move.

- **Two LLM modes behind one pipeline.**
  - *tool mode*: full structured text state + a `submit_move` tool (tests tool
    use + rule understanding).
  - *vision mode*: a rendered PNG + minimal text with per-base occupancy
    withheld, so the model must read the board (tests OCR). Both share parsing,
    retry-with-feedback, heuristic fallback, and telemetry.

- **Data-driven boards.** A terrain is JSON: `nodes` (bases/special bases/HQs),
  `edges` (adjacency + connectivity), `regions` (border bases + medals),
  `medal_objective`. New terrains — and eventually other games — drop in without
  engine changes. Special-base effects are keyed into the `effects.py` registry.

## Extending

- **New terrain:** add `toy_battle/boards/<name>.json`; implement any new
  special-base effect in `effects.py` (`BASE_EFFECTS`).
- **New agent / provider:** subclass `Agent` (or `LLMAgent` and implement
  `_complete`), then register it in `agents/__init__.py` / `agents/llm`.
- **New metric:** add a field to `AgentStats` and surface it in
  `benchmark.aggregate` / `format_summary`.
