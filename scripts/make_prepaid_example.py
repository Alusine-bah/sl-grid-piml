"""Write the prepaid example files in data/example_prepaid/.

    python scripts\\make_prepaid_example.py

A simulated Freetown-style feeder (7 days, 25% of energy stolen, 6 transformer meters) with
44 days of prepaid vending records (37 days before the period plus the period itself) for
about 12,000 customers, 5% of them listed under the wrong transformer.
"""
from pathlib import Path

from slgrid.app_core import simulate_case
from slgrid.io import standard_to_frames
from slgrid.prepaid import VendingConfig, simulate_vending, tokens_to_billing

OUT = Path("data/example_prepaid")


def main():
    d = simulate_case(0.25, 6, "ends", 0.2, False, 7, 11)
    cust, tok = simulate_vending(d, 0, VendingConfig(history_days=44, mapping_error=0.05, seed=12))
    d["billing"] = tokens_to_billing(cust, tok, d["bus_names"], 7)[None]
    d["vending"] = {"customers": cust, "tokens": tok}
    OUT.mkdir(parents=True, exist_ok=True)
    for k, df in standard_to_frames(d).items():
        df.to_csv(OUT / f"{k}.csv", index=False)
    (OUT / "README.md").write_text(
        "# Prepaid example feeder files\n\n"
        "A simulated Freetown-style 11 kV feeder (7 days, 25% of energy stolen, 6 transformer\n"
        "meters) with prepaid vending records instead of billing.csv: `customers.csv` lists\n"
        "about 12,000 meters and the transformer each is linked to (5% linked wrongly), and\n"
        "`tokens.csv` has every token purchase from 37 days before the period to its end\n"
        "(day 0 = first day of head.csv). Made by `scripts/make_prepaid_example.py`.\n")
    print("Wrote", ", ".join(sorted(p.name for p in OUT.iterdir())))


if __name__ == "__main__":
    main()
