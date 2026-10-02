# Data

Downloaded by the pipeline; not committed.

## Source

[Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d), IBM. 7,043 customers,
19 features, 26.5% churn rate.

**All customer data is real.** The retention economics — offer cost, contact cost, acceptance rate,
gross margin and remaining horizon — are declared assumptions, not measurements. They live in
`src/churn/config.py`.

The acceptance outcome per customer does not exist in the data, so `campaign_value` draws it at the
declared rate with a fixed seed. All four decision rules face the same draw, so the comparison is
fair; the absolute dilution figure is not a forecast.
