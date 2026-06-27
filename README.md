# llm-boardgames — Toy Battle

A digital engine for the 1-vs-1 board game **Toy Battle**, plus a framework to
**benchmark LLMs** (OpenAI / Gemini / Anthropic) by having them play it. The
framework probes four things: **OCR / board reading**, **tool use**, **rule
understanding**, and **play skill**. It's built to grow into a multi-game,
multi-model benchmark.

> Mechanics are reimplemented from scratch (game rules aren't copyrightable). No
> original artwork or rulebook text is included; the board is drawn
> programmatically. Toy Battle is a game by Paolo Mori & Alessandro Zucchini
> (Repos Production, 2025) — please support the original.

## Status

- ✅ Full rules engine for Toy Battle (troops, effects, connection, regions, win
  conditions), validated by `random`-vs-`random` self-play that always
  terminates legally.
- ✅ Starter terrain **La plaine des châteaux** (data-driven JSON board).
- ✅ Human CLI, `random` and `heuristic` bots.
- ✅ Match runner + round-robin benchmark (win-rate, medals, illegal-move rate,
  tokens, cost, latency, Elo) with JSONL logs + CSV/JSON summaries.
- ✅ LLM agents for Anthropic / OpenAI / Gemini in **tool** and **vision/OCR**
  modes (plug in API keys via env vars).

## Install

```bash
pip install -e .            # engine + bots + benchmark (+ Pillow for rendering)
pip install -e ".[llm]"     # add the OpenAI / Gemini / Anthropic SDKs
pip install -e ".[dev]"     # add pytest
```

## Play

```bash
# Human vs heuristic bot (prints the board as text and a PNG each turn)
python -m toy_battle.cli play --red human --blue heuristic

# Watch two bots
python -m toy_battle.cli play --red heuristic --blue random --verbose --seed 1

# LLM vs bot (needs the SDK + key, e.g. export ANTHROPIC_API_KEY=...)
python -m toy_battle.cli play --red anthropic:claude-opus-4-8:tool --blue heuristic

# LLM vision/OCR mode vs LLM tool mode
python -m toy_battle.cli play \
  --red openai:gpt-4o:vision \
  --blue gemini:gemini-2.5-pro:tool
```

Agent specs: `random`, `heuristic`, `human`, or
`provider:model[:mode]` where provider ∈ {`anthropic`,`openai`,`gemini`} and
mode ∈ {`tool` (default), `vision`}.

## Benchmark

```bash
# Round-robin; writes per-game JSONL logs + summary.csv/json
python -m toy_battle.cli benchmark --agents random heuristic --games 20 --out results/

# Compare LLMs (tool vs vision), needs keys
python -m toy_battle.cli benchmark \
  --agents anthropic:claude-opus-4-8:tool anthropic:claude-opus-4-8:vision heuristic \
  --games 10 --out results/
```

Example output:

```
Benchmark on plaine_des_chateaux — 20 games
agent            games   win%  medals  illegal%    elo
------------------------------------------------------
heuristic           20 100.0%    6.05     0.00%   1165
random              20   0.0%    0.75     0.00%    835
```

The `illegal%` column (illegal-move attempts / moves) is a useful proxy for an
LLM's rule-understanding / board-reading errors; vision mode tends to raise it
relative to tool mode.

## Other commands

```bash
python -m toy_battle.cli boards                 # list terrains
python -m toy_battle.cli render --out board.png # render the initial board
pytest                                          # run the test suite
```

## How LLMs play

- **Tool mode**: the agent gets the full structured text state and a **numbered
  list of legal moves**, and calls `submit_move(index)`. Tests tool use + rule
  understanding.
- **Vision mode (play with the UI)**: the agent gets **only the rendered board
  image** — no occupancy text and no legal-move list. It must read the board
  itself and name its move in game terms via `submit_move(action, troop, target)`
  (e.g. place `Roxy` on `br_c`), which the engine validates against the rules.
  Tests OCR / board reading.

Both modes also show the **opponent's last action**. Effect sub-decisions
(Cap'taine, Mastok, special bases) are routed back via a `submit_choice` tool.
Parse/illegal responses trigger a retry with feedback, then a heuristic fallback
— all recorded as metrics (`illegal%`). See
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and [`docs/RULES.md`](docs/RULES.md).

## Roadmap

- Remaining 7 terrains and their special-base effects (the engine already
  supports the data + registry).
- A web UI for human play.
- More games behind the same agent/benchmark interface.
