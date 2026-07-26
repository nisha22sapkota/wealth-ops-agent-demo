# Wealth Ops Agent

An AI agent that answers natural-language operations questions over a wealth
management firm's book of business — by calling deterministic tools over real
data, not by generating numbers itself.

This is a working prototype of the idea: *the AI agent that replaces the
analyst hire*, not a chatbot bolted onto a dashboard. Every number in every
answer traces back to a specific source system.

## What it does

Ask a question like:

- *"Which accounts have a tax-loss harvesting opportunity this week?"*
- *"Which households are over-concentrated in a single position?"*

The agent:

1. **Routes** the question to the right deterministic scan (`skills.py`) —
   Claude decides which tool to call and with what parameters, but never
   invents a number itself.
2. **Joins messy multi-source data** — a synthetic custodian export (position
   market values, cost basis) and a portfolio-accounting export (target
   weights, asset class), bridged by an account-mapping table, mirroring how
   a real RIA's data is split across vendor systems.
3. **Writes a decision-ready report**, citing which source system each figure
   came from.
4. **Derives its own chart** from the shape of the actual result — the model
   looks at the real field names and values returned and decides what's worth
   plotting and how (bar vs. no chart, which fields, percent vs. currency),
   rather than the UI having a hardcoded chart template per question type.

## Architecture

```
data/
  custodian_export.csv            # position-level market value / cost basis (messy formatting)
  portfolio_accounting_export.csv # target weights / asset class (different ID scheme)
  account_mapping.csv             # bridges the two systems' account IDs to a household

skills.py   # deterministic pandas joins/scans — the only source of truth for numbers
agent.py    # Claude tool-use loop: routes the question, calls skills.py, writes the
            # report, and derives a ChartSpec from the actual returned data
app.py      # Streamlit chat UI
```

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
streamlit run app.py
```

### Deploying somewhere public

Every question this app answers spends your Anthropic API credits, so if you
deploy it (e.g. Streamlit Community Cloud) with a public URL, also set:

```bash
export APP_PASSWORD=some-shared-code   # gates the whole app behind a code
```

On Streamlit Community Cloud, set `ANTHROPIC_API_KEY` and `APP_PASSWORD` under
**Settings -> Secrets** rather than as shell exports — they're injected as env
vars automatically. Without `APP_PASSWORD` set, the app runs ungated (fine for
local dev, not recommended once the URL is public). As a second layer, each
browser session is capped at `MAX_QUESTIONS_PER_SESSION` (20 by default,
configurable in `app.py`) regardless of the access code.

Or run the agent directly from the CLI:

```bash
python agent.py "Which households are over-concentrated in a single position?"
```

## Why this matters

Every RIA ops team drowns in the same manual work: reconciling custodian data
against portfolio-accounting data, flagging tax-loss harvesting opportunities,
and catching concentration risk — usually by hiring a junior data analyst who
spends months learning the firm's specific systems before producing anything
useful. This agent is a proof of concept that the underlying pipeline —
schema-aware querying, deterministic financial calculations, and decision-ready
reporting — can be automated directly, without requiring a hire to build and
operate it by hand.
