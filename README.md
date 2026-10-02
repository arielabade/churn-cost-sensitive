# Cost-Sensitive Churn

A churn model is not the deliverable. The decision rule is.

---

## 1. Business problem

A telecom wants to spend a retention budget well. It can predict who is likely to leave, but that
alone does not say who to contact: an offer costs money, it only works some of the time, and a
customer it saves is worth their remaining margin, not their monthly bill.

The question is not "who will churn" but "for whom does a retention offer pay for itself".

---

## 2. Key results

**The break-even churn probability is not one number. It ranges 7x across customers.**

```
break-even P(churn) = contact_cost / (acceptance_rate * (value_at_risk - offer_cost))
```

| Percentile | Break-even probability |
| --- | --- |
| 10th | 2.7% |
| Median | 4.1% |
| 90th | 19.6% |

A high-value customer is worth contacting at a 3% churn risk. A low-value one is not worth it below
20%. **No single global cutoff can express that**, which is the whole argument of the project.

### Four ways to pick who gets contacted

Same model, same test set of 1,761 customers. Only the decision rule changes:

| Rule | Contacted | Saved | Net value | vs best |
| --- | --- | --- | --- | --- |
| Threshold at 0.5 (the default) | 450 | 92 | £38,336 | **−21%** |
| Threshold at best F1 (0.18) | 701 | 123 | £48,237 | −0.8% |
| **Per-customer expected value** | 975 | 135 | **£48,621** | — |
| Contact everyone | 1,761 | 146 | £36,950 | −24% |

**The decision this supports.** The 0.5 default — the one that falls out of
`predict()` — leaves **£10,285 on the table, 21% of the achievable value**, by contacting only a
third as many customers as it should. Contacting everyone is worse still: it saves the most customers
and destroys the most value, because the offer cost lands on 1,761 people to retain 146.

**An honest caveat on the headline.** The expected-value rule beats the F1-optimal threshold by
£384, or 0.8%. On this dataset, with these economics, sweeping a threshold for F1 gets you most of
the way. The gap would widen as value dispersion grows — a business with a wider range of customer
values, or a more expensive offer, would see the global cutoff fail harder. The result worth
defending is the comparison against **0.5**, not against a tuned cutoff.

### The model underneath

| | |
| --- | --- |
| ROC AUC | 0.848 |
| Brier | 0.142 |
| Churn rate | 26.5% |

Calibrated with isotonic regression, because the probability is multiplied by money downstream. An
uncalibrated score can rank perfectly and still put the break-even threshold in the wrong place.

---

## 3. Data

| | |
| --- | --- |
| Source | Telco Customer Churn, IBM ([repository](https://github.com/IBM/telco-customer-churn-on-icp4d)) |
| Size | 7,043 customers · 19 features |
| Split | 5,282 train · 1,761 test, stratified |

**All customer data is real. Nothing is simulated.** The retention economics — offer cost,
acceptance rate, margin, horizon — are **declared assumptions**; the dataset publishes none of them.
They live in `config.py` and every result moves with them.

### One cleaning decision worth stating

`TotalCharges` is stored as text and is blank for 11 customers. All 11 have `tenure = 0`: they are
brand new and have not been billed yet. The fill is **0**, not the column mean — filling with the
mean would invent eleven months of billing history for customers who have none, and `tenure` would
then contradict `TotalCharges` in a way a tree model will happily exploit.

---

## 4. Approach

**Why the threshold is per customer.** Solving "expected value of contacting = 0" for the churn
probability leaves `value_at_risk` in the denominator. The threshold therefore depends on the
customer, and customers differ by a factor of seven here. A global cutoff is an averaging of that
away.

**Why the offer cost is weighted by acceptance.** The offer is only paid when it is taken; the
contact is paid regardless. Charging the full offer to everyone contacted overstates the cost and
under-contacts; treating the saved margin as certain overstates the gain and over-contacts. Both are
common, and they fail in opposite directions.

**Why calibration, not just ranking.** The decision multiplies a probability by money. AUC is
invariant to any monotone transform of the scores, so a model can have excellent AUC and place the
break-even threshold badly.

**Why a stratified random split.** The dataset is a snapshot with no date column, so there is no time
to split on. This is a real limitation and is listed as one, not presented as a choice.

---

## 5. Business metrics

```
value_at_risk       = monthly_charges * gross_margin * expected_remaining_months
E[value of contact] = P(churn) * acceptance_rate * value_at_risk
                    - offer_cost * P(churn) * acceptance_rate
                    - contact_cost
break-even P(churn) = contact_cost / (acceptance_rate * (value_at_risk - offer_cost))
```

| Assumption | Value | Why it matters |
| --- | --- | --- |
| Offer cost | £45 | Only charged when accepted. |
| Contact cost | £6 | Charged on everyone contacted. |
| Acceptance rate | 35% | The most optimistic number in any retention business case, and the least often stated. |
| Gross margin | 55% | Revenue is not margin. |
| Remaining horizon | 12 months | Beyond this, the saving is speculative. |

A customer worth less than the offer itself has an **infinite** break-even threshold: saving them
costs more than they are worth. There is a test for it.

---

## 6. Limitations and next steps

- **The acceptance rate is assumed, and it drives everything.** At 35% the campaign nets £48,621. The
  only way to learn the real figure is to hold out a control group and measure it, which no churn
  model can substitute for.
- **This predicts churn, not persuadability.** Some customers would have stayed anyway, and
  contacting them costs money while changing nothing. Some will churn regardless, and the offer is
  wasted on them too. The quantity that actually matters is the **uplift**, and it needs an
  experiment — see [ab-testing-toolkit](https://github.com/arielabade/ab-testing-toolkit).
- **A snapshot, not a time series.** There is no date column, so the model is validated on a random
  split. A production churn model must be validated on a later period than it was trained on.
- **Realised value uses a simulated acceptance draw.** The acceptance outcome per customer is not in
  the data, so `campaign_value` samples it at the declared rate with a fixed seed. The comparison
  between strategies is fair because all four face the same draw; the absolute figure is not a
  forecast.
- **Next step:** make the offer size a decision variable too. Right now every contacted customer gets
  the same £45 offer, when the economics clearly support spending more on high-value customers and
  less on marginal ones.

---

## 7. How to run

```bash
git clone https://github.com/arielabade/churn-cost-sensitive
cd churn-cost-sensitive

python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python -m churn.pipeline   # downloads the data, trains, compares the four rules
pytest                     # 10 tests
```

Change any number in `src/churn/config.py` and re-run: the thresholds, the contact list and the
recommendation all move with it.

### Layout

```
src/churn/   config (the economics), data, economics (the decision rule), pipeline
tests/       the decision rule and the cleaning decisions
reports/     metrics.json, per-customer decisions
```
