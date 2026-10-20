# Data source: Cinelume, a simulated streaming service

The case runs on **Cinelume**, a fictional Brazilian subscription-video service whose operational
database is produced by its own generator (`gerar_dados.py` + `gerador/`, in a separate folder
referred to here as `CINELUME_SOURCE`). Everything in it is synthetic: people, titles, e-mails,
CPFs, IPs. **No real customer appears anywhere in this repository.**

This file records what the data contains, how this project reads it, and what it lacks. It was
written from the source's `README.md`, `schema.sql`, `docs/` and generator entry points, and
from the data itself. The generator's internals (`gerador/simulacao.py`, `expansao.py`) and the
answer key (`_gabarito/`) were deliberately **not** studied before modelling: they encode the
planted patterns, and reading them first would have turned the post-hoc check
([validation](#post-hoc-check-against-the-answer-key)) into a circular exercise.

## Volume used

| | |
| --- | --- |
| Generator | unchanged; default seed **20231001** |
| Scale | **`x60`: 59,952 accounts** (see [why not `full`](#why-60k-accounts-and-not-120k)) |
| History | **37 months**, 2023-10-01 → 2026-10-31 |
| Subscriptions | 68,992 (each reactivation opens a new one) |
| Invoices / payments | 609,658 / 665,081 |
| App sessions | 10,594,094 |
| Plays | 11,151,592 |
| App events | 56,231,976 |
| Generation time | 18.7 min, 3.1 GB peak RSS, on a 16 GB laptop with ~11 GB held by other apps |

Modelling table: **605,245 subscriber-months** (55,278 distinct subscribers × 34 monthly
cut-offs, 2024-01 → 2026-10).

### Why 60k accounts and not 120k

The brief asked for ≥100k subscribers and ≥24 months, or the most that runs in about ten
minutes on an ordinary machine. The history target is met (37 months). The account target is not:

* `--scale full` (≈120k accounts) was run first. It passed 4.1 GB resident memory by month 13
  of 37 and was killed by the machine's out-of-memory daemon (`earlyoom`), with the browser and
  other containers holding ~11 GB.
* The generator offers only `small` (6k), `medium` (30k) and `full` (120k). `medium` gives a
  thin six-month test window (~900 churners a month at 60k becomes ~450).
* **New parameter, no source edit:** `scripts/generate_data.py` registers two extra entries in
  the generator's own `ESCALAS` table at run time (`x60` = 60,000 and `x90` = 90,000 accounts)
  and then calls its unmodified `main()`. Seed, calendar and every other parameter are the
  generator's defaults.
* `x60` ran to completion in 18.7 min. That is above the ten-minute target: the last six
  months alone took 7 min because the machine was swapping. On a machine with ≥8 GB free,
  `make data SCALE=full` is the intended setting and nothing else changes.

```bash
make data SCALE=x60      # what reports/ was built from
make data SCALE=medium   # ~30k accounts, the fast path
make data SCALE=full     # ~120k accounts, needs >6 GB free memory
```

Running the generator writes its answer-key files for that scale into
`CINELUME_SOURCE/_gabarito/`. That is the only change to the source folder.

## How the project reads it

Two interchangeable backends, selected with `--backend`:

| Backend | Business tables | Usage logs |
| --- | --- | --- |
| `parquet` (default) | `data/cinelume/parquet/*.parquet` | same |
| `mysql` | MySQL 8 started from `infra/docker-compose.yml` with the source's `schema.sql` | Parquet (loading 78M log rows into MySQL takes hours) |

`scripts/compare_backends.py` builds the feature table both ways: **605,245 rows × 77 columns,
identical**. The MySQL container is separate from any the source folder may run
(`cinelume-churn-mysql`, port 3317, own volume).

## Tables and grain

29 tables in the schema; the pipeline reads 18. Timestamps are UTC; dates are Brasília civil
days; money is **BRL**.

| Domain | Table | Grain | Used for |
| --- | --- | --- | --- |
| Acquisition | `canais` | channel (6) | acquisition channel |
| | `investimento_diario` | channel × day | not used (CAC is outside the decision) |
| Accounts | `contas` | account | tenure, channel, region, age, marketing consent, staff flag |
| | `perfis` | profile (≤5 per account) | household size, kids, extra members; maps logs to accounts |
| | `dispositivos` | registered device | device count and types; platform for the unit fix |
| Product & price | `planos` | plan (3) | plan code, whether extra members are allowed |
| | `precos` | plan × cycle × validity window (20) | **list price in force at each cut-off**; extra-member fee |
| Subscription | `assinaturas` | continuous subscription period | population, tenure, cycle, **label** |
| | `assinatura_eventos` | lifecycle event | **plan at each cut-off**, upgrades, downgrades, dunning |
| | `pesquisa_cancelamento` | exit-survey answer | not used (written after the outcome) |
| Billing | `faturas` | billing cycle | last invoice, discount in force, days to renewal |
| | `pagamentos` | charge attempt / refund / chargeback | **payment method at each cut-off**, declines, expiries |
| | `cupons`, `cupom_resgates` | coupon; redemption | discount type; **the retention offer (`VOLTA50`)**; staff coupon |
| Usage logs | `sessoes_app` | app session | sessions, days since last session |
| | `reproducoes` | play | hours, active days, titles, completion, buffering, trend |
| | `eventos_app` | UI event | cancellation attempts, plan-screen visits, searches, ratings |
| | `heartbeats_reproducao` | 60 s player ping (sample) | not used (a sample; plays already carry the total) |
| Relationship | `notificacoes` | push / e-mail sent | open rates, re-engagement, price-rise notices, payment-failure e-mails |
| | `tickets_suporte` | support ticket | ticket counts by category, satisfaction |
| | `nps` | survey answer (quarterly sample) | last score |
| Catalogue | `titulos`, `episodios`, … | title, episode, … | not used |

## How churn is defined

The source records two ways out (`assinaturas.tipo_encerramento`):

* **Voluntary** (24,214 subscriptions): the subscriber asks to cancel
  (`cancelamento_solicitado_em`, event `cancelamento_solicitado`) and keeps access until the
  paid cycle ends (`encerrada_em`).
* **Involuntary** (12,134): payment fails after retries (card) or the Pix/boleto expires.

**Target of this project:** `churn = 1` when an active subscriber **requests cancellation within
the calendar month after the cut-off**. That is the decision a retention offer can change, and
it is dated when it is taken, not up to a year later when an annual cycle runs out. Involuntary
churn is kept as `churn_involuntario` for reporting only: a discount does not fix a declined
card; that is a dunning problem.

Rates in the modelling table: voluntary **3.65 %** per subscriber-month, involuntary 1.83 %.

**Population at each cut-off** (first day of the month, 00:00 UTC): subscriptions started before
the cut-off, not ended before it, and **not already pending cancellation**, excluding staff
accounts. One row per account and month.

## Fields available, by theme

| Theme | What exists | How it is used |
| --- | --- | --- |
| Subscription | start, trial end, cancellation request, end, end type, previous subscription | tenure, reactivations, label |
| Plan | plan per subscription **as of 31/10/2026 only**; full history in events | plan rebuilt from events at each cut-off |
| Price | list price per plan × cycle with validity dates; extra-member fee from 10/03/2025; increases on 01/08/2024 and 01/09/2025 | price in force at the cut-off, increase in the last 90 days, monthly revenue at risk |
| Discounts | 14 coupons (%, cycles, validity, channel); redemptions; discount per invoice | discount in force, "next invoice is the first at full price", coupon family |
| Payment | method per attempt, retries D+1/D+3/D+7, decline codes, expiries, refunds, chargebacks | method at the cut-off, declines and expiries in 90 days, dunning state |
| Usage | sessions, plays (seconds, completion, buffering, device), UI events | hours 7/28/84 days, trend, days since last play/session, intent signals |
| Channel | acquisition channel and referrer | channel as feature and reporting segment |
| Dates & events | every table is timestamped; lifecycle events; price-rise notices sent 30 days ahead | the whole temporal design |

## What is missing

* **No randomised retention treatment or control group.** `reengajamento` and `winback` e-mails
  are triggered by rules (inactivity, having left), not by a draw. Uplift — who changes their
  mind *because* of an offer — cannot be estimated, so the acceptance rate stays a declared
  assumption. This is the project's main limitation.
* **No acceptance outcome.** Nobody was offered `VOLTA50` while still subscribed; acceptance is
  simulated, with one draw shared by every rule.
* **No cost side.** No content, streaming or support cost per subscriber: gross margin is
  declared (55 %). No cost per contact.
* **What the data supplies instead of declaring:** plan price at each date, the extra-member fee,
  discounts in force, and **the offer itself** — `VOLTA50`, "50 % off for 2 cycles", priced on
  each customer's own plan (annual subscribers: two monthly-equivalents).

## Imperfections found, and what the pipeline does

The source README warns that the data carries undocumented flaws. These were found from the data:

| Finding | Size | Treatment |
| --- | --- | --- |
| **Watch time in milliseconds.** From app 6.0.0 (Feb 2026) the mobile apps (`mobile_os_a/b`) report most plays in ms in `segundos_assistidos`, documented as seconds. | 2,316,767 plays | Divided by 1000 where the value exceeds 5× the play's wall-clock time (part of the same slice is already in seconds). Left alone, it halved the test-period probabilities (1.9 % predicted vs 3.4 % observed) and the EV rule lost to a tuned F1 cut-off. |
| End-of-data snapshots: `assinaturas.plano_id`, `metodo_pagamento`, `status`; `faturas.status`; `tickets_suporte.status` | all rows | Never ingested. Plan and method rebuilt from history. |
| Device clocks: `eventos_app.recebido_em < ocorrido_em` | 488,495 events | Events dated by `recebido_em` (server): information exists from arrival. |
| `contas.uf` as names or lower case (`São Paulo`, `mg`) | 2,101 accounts | Mapped to the 27 codes, then to region. |
| Staff accounts (`@cinelume.example`; coupon `COLABORADOR`) | 321 / 250 accounts | Excluded from the population. |
| Payment pointing to a missing invoice (`pagamento_id` 664783) | 1 row | MySQL loads with FK checks off so both backends hold the same rows; no feature joins payments to invoices. |
| Plays before their profile was created | 7,405 plays | Profile → account mapping treated as static; no effect on any feature. |
| Accounts created after the documented end (2026-11-01 UTC, still 31/10 in Brasília) | 10 accounts | Outside every cut-off; no effect. |
| Cancellation requests dated after the end of the data | 6 subscriptions | Outside every label window; no effect. |

## Post-hoc check against the answer key

`_gabarito/` holds the simulation's ground truth (latent traits per account, true cause of each
cancellation). It is read only by `churn.validation`, after `reports/` is final, to ask whether
the model recovered the planted patterns. A test fails if any training module names that folder
or imports the validation module; ingestion refuses paths inside it; and a runtime audit hook
blocks Python-level reads of it while the pipeline trains. Results are in the README.
