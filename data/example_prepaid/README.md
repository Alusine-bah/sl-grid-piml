# Prepaid and smart-meter example feeder files

A simulated Freetown-style 11 kV feeder (7 days, 25% of energy stolen, 6 transformer
meters) with prepaid vending records instead of billing.csv:

- `customers.csv`: about 12,000 prepaid meters and the transformer each is linked to
  (5% linked wrongly);
- `tokens.csv`: every token purchase from 37 days before the period to its end
  (day 0 = first day of head.csv);
- `ami_customers.csv` and `ami_readings.csv`: 36 large customers on smart meters at the
  three commercial transformer areas, with hourly readings from 28 days before the
  period (hour 0 = first hour of head.csv).

Made by `scripts/make_prepaid_example.py`.
