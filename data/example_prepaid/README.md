# Prepaid example feeder files

A simulated Freetown-style 11 kV feeder (7 days, 25% of energy stolen, 6 transformer
meters) with prepaid vending records instead of billing.csv: `customers.csv` lists
about 12,000 meters and the transformer each is linked to (5% linked wrongly), and
`tokens.csv` has every token purchase from 37 days before the period to its end
(day 0 = first day of head.csv). Made by `scripts/make_prepaid_example.py`.
