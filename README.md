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
│ 4. AI Content Detector   │   local SLM pipeline (default) / Claude — optional
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
├── detectors/
│   ├── ai_content_detector.py   # AI-text detection: backend selection + Claude path
│   └── local_ai_detector.py     # default backend: offline classifier + SLM reason
├── models/{request,response}.py      # Pydantic I/O (Trust Report schema)
├── cap/cap_wrapper.py           # CAP integration + local simulation
└── data/training_tokens.json    # seed feature vectors for the Isolation Forest
```

---

## Data sources

| Source | Role | Provides |
| --- | --- | --- |
| **GoPlus Token Security** | **Primary** (public, no key) | holder distribution, honeypot, buy/sell tax, mint, blacklist, owner/renounced, hidden-owner, take-back-ownership, anti-whale, LP-lock |
| **Honeypot.is** | honeypot cross-check (simulation, keyless) | independent `is_honeypot` second opinion + buy/sell tax gap-fill |
| **Etherscan V2** (multichain) | verified source + contract age | `source_verified`, `contract_age_days` |
| **DexScreener** | **DEX market data** (primary, no key) | `liquidity_to_mcap_ratio`, `buy_sell_ratio` (aggregated across pools) |
| **GeckoTerminal** | DEX market data (fallback, no key) | same two fields when DexScreener is unavailable |
| **Web3.py** (JSON-RPC) | fallback | name/symbol/decimals/supply, `owner()` |

The GoPlus flags are fed into the model as **raw numeric features** — the agent's
value is *combining* these disconnected signals into one anomaly-aware score, not
re-emitting a single flag as the verdict. Everything degrades gracefully: any
field that can't be fetched becomes `None`, is imputed for the model, and is
reported under `data_quality.missing_fields`.

GoPlus doesn't expose market/liquidity/trading data, so the two market features
(`liquidity_to_mcap_ratio`, `buy_sell_ratio`) are filled from live DEX pool data —
**[DexScreener](https://dexscreener.com)** first, falling back to
**[GeckoTerminal](https://www.geckoterminal.com)** (data courtesy of GeckoTerminal).
Both are free and keyless; if both are unavailable the two fields simply stay
imputed. Filling them raises `data_completeness`, which the completeness-aware
scorer rewards (e.g. DAI moves from 75% / MEDIUM confidence to 85% / HIGH).

**Honeypot cross-check.** GoPlus's `is_honeypot` is static analysis;
**[Honeypot.is](https://honeypot.is)** independently *simulates* a buy + sell as a
second opinion (free, keyless; optional `HONEYPOT_IS_API_KEY` for higher limits).
The two are combined conservatively: if they **disagree**, the token is flagged
`is_honeypot = true` and a `data_quality` note records the conflict; if one source
has no signal, the other fills it. It also gap-fills buy/sell tax from the
simulation when GoPlus didn't provide them. `is_honeypot` stays a single feature —
just set more safely — so no retrain is needed.

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
#   verified-source/age and on-chain reads. AI-content detection defaults to a
#   local, offline model pipeline — see "AI-content detection" below;
#   ANTHROPIC_API_KEY is only needed with AI_DETECTOR_BACKEND=anthropic.
```

### Run the API

```bash
python app.py
# or: uvicorn app:app --reload
```

- **http://localhost:8000/ui** — the **web dashboard** (paste an address → visual Trust Report)
- **http://localhost:8000/docs** — Swagger UI

The Isolation Forest trains (or loads) on startup, so the first request is fast.

### Web UI

A dependency-free, single-file dashboard ([`web/index.html`](web/index.html),
served at **`/ui`**) for demoing the analyzer: enter a token address, pick the
chain, and see a color-coded risk gauge, the confidence/data-completeness line,
flags, key metrics, the explainable score breakdown, and the AI-content check —
all rendered from the same-origin `POST /analyze` response (no keys, no
third-party frontend calls).

---

## Run with Docker

Build once, then run the whole pipeline in a clean container — no local Python
setup, reproducible for the demo/deploy.

```bash
docker build -t token-trust-analyzer .
docker run -p 8000:8000 token-trust-analyzer
```

Then open **http://localhost:8000/ui** (dashboard) and
**http://localhost:8000/docs** (Swagger). On startup the container fits the
Isolation Forest from `data/training_tokens.json` (visible in the log); once it's
ready, `GET /health` returns `status: ok`.

**Keys are passed at runtime, never baked into the image.** GoPlus is keyless, so
the container runs with no config at all; add the optional/recommended vars (see
[Environment variables](#environment-variables)) via `--env-file` or `-e`:

```bash
docker run -p 8000:8000 --env-file .env token-trust-analyzer
```

Or with Compose (reads `.env` if present):

```bash
docker compose up --build
```

The image is a lean `python:3.12-slim` (production deps only — the test suite and
`requirements-dev.txt` are not shipped) and runs as a non-root user. The trained
model isn't included: it refits fresh on first boot from the training data that
ships in the image.

---

## HTTP API

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/analyze` | Full pipeline: collect → features → score → (AI detect) → report (with a `cached` flag) |
| `POST` | `/analyze/batch` | Analyze up to 25 tokens in one request (deduped, bounded concurrency, per-token error isolation) |
| `POST` | `/score` | ML scoring only from a pre-built feature set (**no API keys / network needed**) |
| `POST` | `/detect-ai` | AI-generated-content detection only (local SLM pipeline by default; see `AI_DETECTOR_BACKEND`) |
| `POST` | `/cap/analyze` | Run `/analyze` inside a simulated CAP payment cycle |
| `GET`  | `/health` | Liveness + which data sources are configured |

### Caching & batch

The expensive on-chain collection (GoPlus / Etherscan / Web3 / DEX) is cached
**in-memory, per-process**, keyed by `(chain, address)`; scoring and AI detection
still run fresh on top, so `project_text` is always honored. A repeated `/analyze`
returns with `"cached": true` and skips the network (≈50× faster in practice). TTL
is set by **`CACHE_TTL_SECONDS`** (default `600`); **`CACHE_TTL_SECONDS=0` disables
caching entirely** (useful for fresh data / tests).

`POST /analyze/batch` takes `{ "tokens": [{ "contract_address", "chain" }, …],
"project_text"? }` and returns `{ "results": [{ "contract_address", "chain",
"report" | "error" }, …] }` — one entry per token, in order. Duplicate tokens are
analyzed once; a bad address or failing lookup for one token yields an `error`
entry without failing the batch; batches over 25 tokens are rejected with `422`.
Per-token pipelines run with bounded concurrency (`BATCH_CONCURRENCY`, default 5)
to respect the upstream rate limits.

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

## AI-content detection (local by default)

The optional AI-text check (`/detect-ai`, and `/analyze` when `project_text` /
`project_url` is supplied) is **pluggable** via `AI_DETECTOR_BACKEND`:

| Backend | What runs | Needs |
| --- | --- | --- |
| `local` (default) | Offline two-model pipeline: a RoBERTa-style **classifier** decides `is_ai_generated` + `confidence`, then a small instruct **SLM** writes the one–two-sentence `reason`. No API calls, no cost. | `pip install -r requirements-slm.txt` |
| `anthropic` | The original Claude path (`claude-sonnet-4-6`). | `ANTHROPIC_API_KEY` |
| `off` | Detection disabled — always `checked: false`. | — |

All backends return the same shape (`checked`, `is_ai_generated`, `confidence`,
`reason`, `source`); the local backend sets `source` to the models used, e.g.
`"local:chatgpt-detector-roberta+qwen2.5-1.5b-instruct"`.

### Enabling local mode

```bash
pip install -r requirements-slm.txt   # transformers + torch (kept OUT of requirements.txt)
# CPU-only torch is much smaller than the default CUDA build:
#   pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Notes / caveats:

- **First detection downloads model weights** from the Hugging Face Hub
  (~0.5 GB classifier + ~3 GB `Qwen/Qwen2.5-1.5B-Instruct`) into
  `~/.cache/huggingface`; afterwards it is fully offline. Models are
  **lazy-loaded on first use** — startup stays instant, and the base app deploys
  fine without the SLM deps installed.
- **CPU latency:** the classifier verdict is fast (<1 s); the SLM reason takes
  roughly 5–30 s on CPU depending on hardware. For snappier reasons set
  `AI_DETECTOR_SLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct`.
- **Graceful degradation:** if `transformers`/`torch` aren't installed or the
  classifier can't load, the result is `checked: false` with a clear reason —
  never a crash. If only the reason-SLM fails, the classifier's verdict is kept
  with a templated reason (and `source` names just the classifier).
- The reason is generated with greedy decoding (`do_sample=False`), so it is
  deterministic for a given text.
- **Docker:** the default image stays lean and does **not** include the SLM
  deps or weights. The opt-in is a build arg —
  `docker build --build-arg INSTALL_SLM=true -t token-trust-analyzer .` —
  which installs CPU-only torch + `requirements-slm.txt` into the image (see
  the `Dockerfile`). Mount a Hugging Face cache volume
  (`-v hf-cache:/home/appuser/.cache/huggingface`) so weights download once,
  not on every container start.

Live smoke test (not part of CI — it downloads real weights):

```bash
pip install -r requirements-slm.txt
AI_DETECTOR_BACKEND=local python app.py &
curl -X POST http://localhost:8000/detect-ai -H 'Content-Type: application/json' \
  -d '{"project_text":"Our revolutionary synergistic web3 protocol leverages cutting-edge blockchain technology to empower communities worldwide..."}'
# → {"checked": true, "is_ai_generated": ..., "confidence": ...,
#    "reason": "<SLM-written sentence>", "source": "local:chatgpt-detector-roberta+qwen2.5-1.5b-instruct"}
# No Anthropic call is made; unset ANTHROPIC_API_KEY to prove it.
```

---

## Running tests

The suite is **fully offline** — every external call (GoPlus, Etherscan, Web3,
Anthropic, CROO) is mocked, and the local AI-detector models are monkeypatched
(no `transformers`/`torch` needed, no weight downloads), so it passes on a clean
checkout with **no API keys**.

```bash
pip install -r requirements-dev.txt
pytest
```

It covers the rules, feature extractor, hybrid scorer (including the
completeness weighting), the GoPlus/Etherscan collector, the AI-content detector
(the local SLM backend, the Claude path, and `AI_DETECTOR_BACKEND` switching),
and every HTTP endpoint via `TestClient`. GitHub Actions runs it on Python 3.11
and 3.12 for every push and pull request (`.github/workflows/tests.yml`).

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

### A2A composability — a consumer agent hires this one

The real marketplace value is **agent-to-agent (A2A)**: another agent can *hire*
this analyzer over CAP, pay in USDC on Base, and consume the delivered Trust
Report with no human in the loop. [`cap/consumer.py`](cap/consumer.py) is that
**requester** agent — the mirror image of the provider above:

```
negotiate_order(requirements={contract_address, chain})    # POST
        │  provider accepts → Order created
        ▼
pay_order(order_id)                                        # LOCK  (USDC escrowed on Base)
        │  provider runs the pipeline → deliver_order
        ▼
await ORDER_COMPLETED → get_delivery(order_id)             # DELIVER + CLEAR
        │
        ▼  parse the Trust Report JSON → print score / risk / flags
```

**One-command demo (no keys, no funded wallet):**

```bash
./scripts/demo_a2a.sh
# or: python -m cap.consumer 0x6B175474E89094C44Da98b954EedeAC495271d0F
```

With no `CROO_*` env set it runs in **`[SIMULATION]`** mode: it hires the *local*
analyzer (`POST /analyze` if a server is up, else the pipeline in-process) and
narrates the same Post → Lock → Deliver → Clear steps, ending with the Trust
Report summary — this is what the demo video shows end-to-end without a wallet.

**Live A2A over CROO** (`[LIVE CROO]`, real on-chain settlement) uses two
identities and a small USDC balance on Base. Run the provider worker
(`python -m cap.cap_wrapper`), then run the consumer with its **own** key:

```bash
CONSUMER_CROO_SDK_KEY=croo_sk_<buyer> \
CROO_SERVICE_ID=<analyzer service id> \
python -m cap.consumer 0x6B175474E89094C44Da98b954EedeAC495271d0F
```

The consumer negotiates → pays → waits for delivery → prints the report, logging
each CAP event (negotiation created → order accepted → paid → delivered →
completed) as it happens. It signs as `CONSUMER_CROO_SDK_KEY` (falling back to
`CROO_SDK_KEY`), so the requester is a distinct marketplace identity.

---

## Environment variables

| Var | Required | Purpose |
| --- | --- | --- |
| `GOPLUS_API_KEY` | — | GoPlus works keyless; set only to raise rate limits. |
| `HONEYPOT_IS_API_KEY` | — | Honeypot.is works keyless; optional key raises rate limits (`X-API-KEY`). |
| `ETHERSCAN_API_KEY` | recommended | Verified-source + contract age (Etherscan V2). |
| `WEB3_RPC_URL` | recommended | JSON-RPC endpoint (direct ERC-20 reads, `owner()`). |
| `CHAIN` | — | Default chain (`ethereum` \| `base`). |
| `AI_DETECTOR_BACKEND` | — | AI-content detection backend: `local` (default; offline two-model pipeline) \| `anthropic` (Claude) \| `off`. |
| `AI_DETECTOR_CLASSIFIER_MODEL` | — | Local classifier (HF id); defaults to `Hello-SimpleAI/chatgpt-detector-roberta`. |
| `AI_DETECTOR_SLM_MODEL` | — | Local reason-generator (HF id); defaults to `Qwen/Qwen2.5-1.5B-Instruct`. |
| `ANTHROPIC_API_KEY` | for `anthropic` backend | Claude API key (AI-content detection). |
| `ANTHROPIC_MODEL` | — | Defaults to `claude-sonnet-4-6`. |
| `CROO_SDK_KEY` | for live CAP | SDK key (`croo_sk_...`). Blank → simulation. |
| `CONSUMER_CROO_SDK_KEY` | for live A2A | Requester identity for `cap/consumer.py`; falls back to `CROO_SDK_KEY`. |
| `CROO_API_URL` | for live CAP | CROO API base URL. |
| `CROO_WS_URL` | for live CAP | CROO websocket URL. |
| `BASE_RPC_URL` | — | Base chain RPC (on-chain reads / settlement). |
| `CROO_SERVICE_ID` | for live CAP | The service id created on the Store. |
| `CROO_WALLET_ADDRESS` | — | Provider wallet (settlement / logging). |
| `CACHE_TTL_SECONDS` | — | Analyze cache TTL in seconds (default `600`; `0` disables caching). |
| `CACHE_MAX_ENTRIES` | — | Max cached tokens before LRU eviction (default `512`). |
| `BATCH_CONCURRENCY` | — | Max concurrent per-token analyses in `/analyze/batch` (default `5`). |
| `HOST`, `PORT`, `LOG_LEVEL` | — | Local server config. |
| `APP_BASE_URL` | — | Where the consumer's simulation POSTs `/analyze` (default `http://localhost:8000`). |

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
