"""Write the prepaid + smart-meter example files in data/example_prepaid/.

    python scripts\\make_prepaid_example.py

A simulated Freetown-style feeder (7 days, 25% of energy stolen, 6 transformer meters):
  * prepaid vending records for about 12,000 customers: 44 days of token purchases
    (37 days before the period plus the period), 5% of meters linked to the wrong transformer;
  * 36 large customers at the three commercial transformer areas on smart meters, with
    28 days of hourly history before the period.
"""
from pathlib import Path

from slgrid.app_core import simulate_case
from slgrid.io import standard_to_frames

OUT = Path("data/example_prepaid")


def main():
    d = simulate_case(0.25, 6, "ends", 0.2, False, 7, 11, billing="prepaid", mapping_error=0.05,
                      ami=True, vending_history=44)
    OUT.mkdir(parents=True, exist_ok=True)
    for k, df in standard_to_frames(d).items():
        df.to_csv(OUT / f"{k}.csv", index=False)
    (OUT / "README.md").write_text(
        "# Prepaid and smart-meter example feeder files\n\n"
        "A simulated Freetown-style 11 kV feeder (7 days, 25% of energy stolen, 6 transformer\n"
        "meters) with prepaid vending records instead of billing.csv:\n\n"
        "- `customers.csv`: about 12,000 prepaid meters and the transformer each is linked to\n"
        "  (5% linked wrongly);\n"
        "- `tokens.csv`: every token purchase from 37 days before the period to its end\n"
        "  (day 0 = first day of head.csv);\n"
        "- `ami_customers.csv` and `ami_readings.csv`: 36 large customers on smart meters at the\n"
        "  three commercial transformer areas, with hourly readings from 28 days before the\n"
        "  period (hour 0 = first hour of head.csv).\n\n"
        "Made by `scripts/make_prepaid_example.py`.\n")
    print("Wrote", ", ".join(sorted(p.name for p in OUT.iterdir())))


if __name__ == "__main__":
    main()
