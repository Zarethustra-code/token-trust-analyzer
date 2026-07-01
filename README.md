# Token Trust Analyzer

An explainable AI agent that scores the **trustworthiness of an ERC-20 token**.
Give it a contract address; it returns a structured **Trust Report** — a 0–100
risk score, a LOW/MEDIUM/HIGH level, human-readable flags, key metrics, and a
plain-language explanation of *why* the score is what it is.

Built for the **CROO AI Agent Hackathon**. It is deployed on the **CROO Agent
Protocol (CAP)** as a paid, composable agent that settles on-chain.

- **Input:** an ERC-20 contract address (Ethereum or Base)
- **Output:** a structured, explainable Trust Report (JSON)
- **ML:** hybrid — interpretable rules **+ an Isolation Forest anomaly detector** (the centerpiece), with an optional XGBoost layer
- **Settlement:** USDC on **Base**, via CAP
- **License:** MIT

---

## What makes the ML real (not if-else in a trenchcoat)

The score is a **blend of three signals**, and every point is traceable to a feature:

```
final_score = clamp(
      W_ANOMALY   * anomaly_score        # Isolation Forest, 0–100 (the ML centerpiece)
    + rule_penalty_total                 # interpretable per-feature heuristics
    + W_SUPERVISED * supervised_prob*100 # XGBoost P(scam), 0 if no model is loaded
, 0, 100)
```

- **Isolation Forest** (`ml/anomaly_model.py`) trains on a seed set of *healthy*
  tokens (`data/training_tokens.json`) and learns what "normal" looks like. A new
  token whose on-chain profile sits far from that manifold scores as an outlier.
  The raw sklearn score is calibrated onto 0–100 (median-normal → ~0, ≥95th
  percentile of normal → ~100). The fitted model is cached with joblib and
  trained lazily on first run — no separate training step needed.
- **Rules** (`ml/rules.py`) add interpretable penalties (honeypot +40, liquidity
  not locked +30, whale > 50% +25, unverified source +20, hidden owner +20, …).
  They **skip unknown features** — a missing value is never penalized.
- **XGBoost** (`ml/supervised_model.py`) is a graceful no-op until a labeled
  dataset is provided; then it contributes `P(scam)`.

Every flag in the report carries its `rule`, `points`, and the `feature` it
inspected, so the number is fully auditable.

### The 20 features

`top_holder_pct`, `top10_holder_pct`, `holder_count`, `gini`, `creator_percent`,
`liquidity_locked`, `liquidity_to_mcap_ratio`, `source_verified`, `has_mint`,
`ownership_renounced`, `has_blacklist`, `is_honeypot`, `buy_tax`, `sell_tax`,
`hidden_owner`, `can_take_back_ownership`, `is_anti_whale`, `contract_age_days`,
`recent_tx_count`, `buy_sell_ratio`.

Missing values are imputed to a **healthy-token prior** so that *absence of data
is treated neutrally* by the model (the rule engine, conversely, only penalizes
confirmed risks). `gini` is derived from the holder distribution.

### Rebuilding the training set

The Isolation Forest learns "normal" from `data/training_tokens.json`. To
regenerate it with a larger, more diverse set of *healthy* tokens:

```bash
python scripts/build_training_set.py --target 200
```

This pulls a curated list of well-known, legitimate ERC-20s (Ethereum + Base)
through the **same** collector + extractor the app uses at inference time, and
verifies each token's returned symbol against its expected symbol (mismatches are
dropped, so a mistyped address can't pollute the set). It then tops up to
`--target` with **synthetic** healthy rows sampled from the seed distribution
(`data/seed_tokens.json`). Offline, or to skip the network, use
`--synthetic-only` — the whole set is then generated synthetically. The script
refreshes the cached model so the Isolation Forest refits on the next app start.
A broader set markedly reduces false anomalies on legitimate tokens (e.g. DAI's
anomaly score dropped from ~69/100 on the original 45-row seed to ~0 on the
expanded set).

---

## Architecture

```
Token Contract Address
        │
        ▼
┌──────────────────────────┐   GoPlus (primary) + Etherscan + Web3
│ 1. On-Chain Collector    │   graceful degradation: any field → None
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│ 2. Feature Extractor     │   raw data → 20-feature vector (+ Gini, imputation)
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│ 3. Hybrid ML Scorer      │   rules + Isolation Forest (+ optional XGBoost)
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│ 4. AI Content Detector   │   Claude (claude-sonnet-4-6) — optional
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│ 5. Report Generator      │   structured Trust Report (Pydantic / JSON)
└────────────┬─────────────┘
             ▼
        FastAPI + CROO CAP wrapper
```

```
token-trust-analyzer/
├── app.py                       # FastAPI entry point — wires everything together
├── collectors/onchain_collector.py   # GoPlus (primary) + Etherscan + Web3
├── features/feature_extractor.py     # raw → 20-feature vector, Gini, imputation
├── ml/
│   ├── rules.py                 # interpretable heuristic penalties
│   ├── anomaly_model.py         # Isolation Forest (train + calibrated score)
│   ├── supervised_model.py      # optional XGBoost (graceful no-op)
│   └── scorer.py                # blends the three signals + explanation
├── detectors/ai_content_detector.py  # Claude AI-text detection (optional)
├── models/{request,response}.py      # Pydantic I/O (Trust Report schema)
├── cap/cap_wrapper.py           # CAP integration + local simulation
└── data/training_tokens.json    # seed feature vectors for the Isolation Forest
```

---

## Data sources

| Source | Role | Provides |
| --- | --- | --- |
| **GoPlus Token Security** | **Primary** (public, no key) | holder distribution, honeypot, buy/sell tax, mint, blacklist, owner/renounced, hidden-owner, take-back-ownership, anti-whale, LP-lock |
| **Etherscan V2** (multichain) | verified source + contract age | `source_verified`, `contract_age_days` |
| **Web3.py** (JSON-RPC) | fallback | name/symbol/decimals/supply, `owner()` |

The GoPlus flags are fed into the model as **raw numeric features** — the agent's
value is *combining* these disconnected signals into one anomaly-aware score, not
re-emitting a single flag as the verdict. Everything degrades gracefully: any
field that can't be fetched becomes `None`, is imputed for the model, and is
reported under `data_quality.missing_fields`.

---

## Setup

**Requirements:** Python 3.11+

```bash
cd token-trust-analyzer

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
#   GoPlus works with no key. Add ETHERSCAN_API_KEY + WEB3_RPC_URL for
#   verified-source/age and on-chain reads. ANTHROPIC_API_KEY only for /detect-ai.
```

### Run the API

```bash
python app.py
# or: uvicorn app:app --reload
```

- **http://localhost:8000/docs** — Swagger UI

The Isolation Forest trains (or loads) on startup, so the first request is fast.

---

## HTTP API

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/analyze` | Full pipeline: collect → features → score → (AI detect) → report |
| `POST` | `/score` | ML scoring only from a pre-built feature set (**no API keys / network needed**) |
| `POST` | `/detect-ai` | AI-generated-content detection only (Claude) |
| `POST` | `/cap/analyze` | Run `/analyze` inside a simulated CAP payment cycle |
| `GET`  | `/health` | Liveness + which data sources are configured |

### `/analyze`

```jsonc
// request
{ "contract_address": "0x6B175474E89094C44Da98b954EedeAC495271d0F",
  "chain": "ethereum",            // ethereum | base (optional, default ethereum)
  "project_text": "optional marketing text for AI detection" }
```

```jsonc
// response (Trust Report, abridged)
{ "contract_address": "0x6B17...", "chain": "ethereum",
  "token": { "name": "Dai", "symbol": "DAI", "decimals": 18, "total_supply": ... },
  "trust_score": 12, "risk_level": "LOW",
  "flags": [],
  "metrics": { "top_holder_pct": 5.2, "is_honeypot": false, "sell_tax": 0.0, ... },
  "score_breakdown": { "rule_penalties": [...], "anomaly_score": 8.1,
                       "anomaly_contribution": 2.8, "supervised_prob": null, ... },
  "ai_generated_content": { "checked": false, "reason": "No project text ..." },
  "data_quality": { "sources_used": ["goplus","etherscan","web3"], "missing_fields": [...] },
  "explanation": "Risk score 12/100 (LOW). ...",
  "generated_at": "..." }
```

```bash
curl -X POST http://localhost:8000/analyze \
  -H 'Content-Type: application/json' \
  -d '{"contract_address":"0x6B175474E89094C44Da98b954EedeAC495271d0F","chain":"ethereum"}'
```

### `/score` — offline demo (no keys)

Send raw features directly; missing keys are imputed. Great for demoing the ML
without any network access:

```bash
curl -X POST http://localhost:8000/score -H 'Content-Type: application/json' -d '{
  "features": {"is_honeypot": 1, "liquidity_locked": 0, "top_holder_pct": 72,
               "sell_tax": 55, "source_verified": 1, "contract_age_days": 1}
}'
# → HIGH, with each fired rule tied to its feature
```

---

## CAP integration (the on-chain part)

CAP is **event-driven** and CROO controls settlement — the provider reacts to
events. Registration and **service pricing are configured on the CROO Agent Store
dashboard**, not in code (there is **no** `register_agent()`); the code only
references a `service_id`.

**Provider lifecycle (`cap/cap_wrapper.py`):**

```
connect_websocket()                       # SDK already starts the read/ping loops
  ├─ NEGOTIATION_CREATED  ── buyer wants the service
  │     ├─ get_negotiation(e.negotiation_id) → parse .requirements (contract_address)
  │     └─ accept_negotiation(e.negotiation_id) → Order created
  │          (buyer request cached by result.order.order_id)
  ├─ ORDER_PAID           ── buyer's USDC is escrow-locked on Base
  │     └─ run the pipeline → deliver_order(e.order_id, DeliverOrderRequest(
  │            deliverable_type=DeliverableType.TEXT, deliverable_text=<report JSON>))
  └─ ORDER_COMPLETED      ── delivery accepted → escrow clears / settles on Base
```

This maps to **Post → Lock → Deliver → Clear**. The buyer's inputs (contract
address, chain, optional project text) arrive on the **Negotiation** — the code
fetches them via `get_negotiation()` and caches the parsed request by
`order_id`, then runs the pipeline when the order is paid.

**Run the provider worker (real, on-chain):**

```bash
python -m cap.cap_wrapper
```

With `CROO_SDK_KEY` + `CROO_API_URL` + `CROO_WS_URL` set (and `croo-sdk`
installed) it serves real orders. **Without** them it runs a local
**Post→Lock→Deliver→Clear simulation** so you can develop offline. The same
simulation backs `POST /cap/analyze`.

> The SDK surface (`AgentClient`, `Config`, `EventType`, `DeliverableType`,
> `DeliverOrderRequest`, `connect_websocket`, `get_negotiation`,
> `accept_negotiation`, `get_order`, `reject_order`, `deliver_order`) was verified
> against `croo-sdk` 0.2.1 / `examples/provider.py`. The local package is named
> `cap/` (not `croo/`) so it doesn't shadow the installed SDK.

### To go live

1. On the **CROO Agent Store dashboard**: register the agent, price the service, note its `service_id`.
2. Fill the `CROO_*` variables in `.env` (including the required `CROO_WS_URL`).
3. `pip install 'croo-sdk==0.2.1'` (already in `requirements.txt`).
4. `python -m cap.cap_wrapper`.

---

## Environment variables

| Var | Required | Purpose |
| --- | --- | --- |
| `GOPLUS_API_KEY` | — | GoPlus works keyless; set only to raise rate limits. |
| `ETHERSCAN_API_KEY` | recommended | Verified-source + contract age (Etherscan V2). |
| `WEB3_RPC_URL` | recommended | JSON-RPC endpoint (direct ERC-20 reads, `owner()`). |
| `CHAIN` | — | Default chain (`ethereum` \| `base`). |
| `ANTHROPIC_API_KEY` | for `/detect-ai` | Claude API key (AI-content detection). |
| `ANTHROPIC_MODEL` | — | Defaults to `claude-sonnet-4-6`. |
| `CROO_SDK_KEY` | for live CAP | SDK key (`croo_sk_...`). Blank → simulation. |
| `CROO_API_URL` | for live CAP | CROO API base URL. |
| `CROO_WS_URL` | for live CAP | CROO websocket URL. |
| `BASE_RPC_URL` | — | Base chain RPC (on-chain reads / settlement). |
| `CROO_SERVICE_ID` | for live CAP | The service id created on the Store. |
| `CROO_WALLET_ADDRESS` | — | Provider wallet (settlement / logging). |
| `HOST`, `PORT`, `LOG_LEVEL` | — | Local server config. |

---

## Notes & design choices

- **Explainable by construction.** Flags carry `rule` + `points` + `feature`;
  `score_breakdown` shows exactly how the number was assembled.
- **Graceful degradation everywhere.** Missing data → `None` → imputed → reported
  in `data_quality`; the request never crashes.
- **Runs without CROO and without a supervised model.** Phase-1 scoring works with
  only GoPlus (keyless); `/score` needs no network at all.
- **`# TODO(CAP)` markers** flag the handful of CAP payload shapes to confirm
  against the live SDK.

## License

[MIT](LICENSE)
