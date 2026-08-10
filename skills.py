"""
Deterministic analysis skills over messy multi-source wealth management data.

Two data sources are joined here on purpose: a custodian export (position-level
market values and cost basis, one row per lot-like holding) and a portfolio
accounting export (target weights / asset class, keyed by a different account
ID scheme). A mapping table bridges the two. This mirrors how a real RIA's
data is fragmented across custodian (Schwab/Fidelity/Pershing-style) and
portfolio accounting (Orion/Black Diamond/Addepar-style) systems.

Every function returns plain data (list of dicts) with a "source" field per
row, rather than prose — the LLM layer in agent.py turns this into a
decision-ready answer, but the numbers themselves come only from these
deterministic joins, never from the model.
"""

import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _money(series: pd.Series) -> pd.Series:
    """Parse messy currency strings like '$28,113.00' into floats."""
    return (
        series.astype(str)
        .str.replace(r"[$,]", "", regex=True)
        .astype(float)
    )


def _load_custodian() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(DATA_DIR, "custodian_export.csv"))
    df["MarketValue"] = _money(df["MarketValue"])
    df["CostBasis"] = _money(df["CostBasis"])
    df["AcquiredDate"] = pd.to_datetime(df["AcquiredDate"], format="%m/%d/%Y")
    return df


def _load_mapping() -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_DIR, "account_mapping.csv"))


def _load_portfolio_accounting() -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_DIR, "portfolio_accounting_export.csv"))


def _load_transactions() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(DATA_DIR, "transactions.csv"))
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    return df


def _custodian_with_household() -> pd.DataFrame:
    custodian = _load_custodian()
    mapping = _load_mapping()
    merged = custodian.merge(
        mapping,
        left_on="AccountNumber",
        right_on="custodian_account_number",
        how="left",
    )
    return merged


def tlh_scan(threshold_pct: float = 10.0) -> list[dict]:
    """
    Find positions with an unrealized loss beyond threshold_pct, using
    custodian market value vs. cost basis (the only system of record for
    realized/unrealized gain-loss).
    """
    df = _custodian_with_household()
    df["unrealized_gain_loss"] = df["MarketValue"] - df["CostBasis"]
    df["gain_loss_pct"] = (df["unrealized_gain_loss"] / df["CostBasis"]) * 100

    hits = df[df["gain_loss_pct"] <= -threshold_pct].copy()
    hits = hits.sort_values("gain_loss_pct")

    return [
        {
            "household": row["household_name"],
            "account": row["AccountNumber"],
            "advisor": row["advisor"],
            "symbol": row["Symbol"],
            "market_value": round(row["MarketValue"], 2),
            "cost_basis": round(row["CostBasis"], 2),
            "unrealized_loss": round(row["unrealized_gain_loss"], 2),
            "loss_pct": round(row["gain_loss_pct"], 1),
            "acquired_date": row["AcquiredDate"].strftime("%Y-%m-%d"),
            "source": "custodian_export.csv (position-level cost basis)",
        }
        for _, row in hits.iterrows()
    ]


def concentration_scan(threshold_pct: float = 20.0) -> list[dict]:
    """
    Find single-position concentration risk at the HOUSEHOLD level (summed
    across every account in the household, via account_mapping.csv), using
    custodian market values as the source of truth for position size.
    """
    df = _custodian_with_household()

    household_totals = df.groupby("household_name")["MarketValue"].sum()
    position_totals = df.groupby(["household_name", "Symbol"])["MarketValue"].sum()

    hits = []
    for (household, symbol), value in position_totals.items():
        total = household_totals[household]
        weight_pct = (value / total) * 100
        if weight_pct >= threshold_pct:
            accounts = sorted(
                df[(df["household_name"] == household) & (df["Symbol"] == symbol)][
                    "AccountNumber"
                ]
                .unique()
                .tolist()
            )
            hits.append(
                {
                    "household": household,
                    "symbol": symbol,
                    "position_value": round(value, 2),
                    "household_total_value": round(total, 2),
                    "weight_pct": round(weight_pct, 1),
                    "accounts": accounts,
                    "source": (
                        "custodian_export.csv (position market values) joined to "
                        "account_mapping.csv (household rollup)"
                    ),
                }
            )

    hits.sort(key=lambda h: -h["weight_pct"])
    return hits


def wash_sale_scan(threshold_pct: float = 10.0, as_of: str | None = None) -> list[dict]:
    """
    For every unrealized-loss position that tlh_scan would flag as a harvesting
    candidate, check whether harvesting it today would trigger a wash sale --
    a purchase of the same symbol within the trailing 30 days in ANY account
    belonging to the same household. This deliberately looks across the whole
    household (every account_type and every owner, including a spouse's
    account and IRA/Roth accounts), not just the account holding the loss --
    IRS wash-sale attribution applies across spouses and across a person's own
    IRA even though those accounts don't share a 1099 with the taxable
    account. A single-account or single-custodian tool cannot see this.

    Args:
        threshold_pct: Same meaning as in tlh_scan -- minimum unrealized loss,
            as a positive percent, to be considered a harvesting candidate.
        as_of: ISO date (YYYY-MM-DD) to treat as "today" for the 30-day
            lookback window. Defaults to the current date. Pass this
            explicitly for reproducible results against this demo's static
            transactions.csv dates.
    """
    as_of_ts = pd.Timestamp(as_of) if as_of else pd.Timestamp.now().normalize()
    window_start = as_of_ts - pd.Timedelta(days=30)

    holdings = _custodian_with_household()
    holdings["unrealized_gain_loss"] = holdings["MarketValue"] - holdings["CostBasis"]
    holdings["gain_loss_pct"] = (holdings["unrealized_gain_loss"] / holdings["CostBasis"]) * 100
    loss_candidates = holdings[holdings["gain_loss_pct"] <= -threshold_pct]

    txns = _load_transactions().merge(
        _load_mapping(),
        left_on="AccountNumber",
        right_on="custodian_account_number",
        how="left",
    )
    recent_buys = txns[
        (txns["Action"] == "BUY")
        & (txns["Date"] >= window_start)
        & (txns["Date"] <= as_of_ts)
    ]

    hits = []
    for _, row in loss_candidates.iterrows():
        conflicts = recent_buys[
            (recent_buys["household_name"] == row["household_name"])
            & (recent_buys["Symbol"] == row["Symbol"])
        ]
        if conflicts.empty:
            continue

        conflict_rows = [
            {
                "purchase_account": c["AccountNumber"],
                "owner": c["owner"],
                "account_type": c["account_type"],
                "purchase_date": c["Date"].strftime("%Y-%m-%d"),
                "quantity": int(c["Quantity"]),
            }
            for _, c in conflicts.iterrows()
        ]
        ira_involved = any(c["account_type"] in ("IRA", "Roth") for c in conflict_rows)

        hits.append(
            {
                "household": row["household_name"],
                "symbol": row["Symbol"],
                "loss_account": row["AccountNumber"],
                "loss_owner": row["owner"],
                "loss_account_type": row["account_type"],
                "unrealized_loss": round(row["unrealized_gain_loss"], 2),
                "loss_pct": round(row["gain_loss_pct"], 1),
                "conflicting_purchases": conflict_rows,
                "risk": (
                    "Repurchase occurred in an IRA/Roth -- if harvested, the loss "
                    "is PERMANENTLY disallowed (no basis step-up is available in "
                    "a tax-deferred account)."
                    if ira_involved
                    else "Repurchase occurred in a taxable account -- the loss is "
                    "deferred and added to the replacement lot's basis, not "
                    "permanently lost, but harvesting now will not currently "
                    "reduce taxable income."
                ),
                "source": (
                    "custodian_export.csv (loss position) joined to "
                    "transactions.csv (recent household-wide purchases) via "
                    "account_mapping.csv (household + owner + account_type rollup)"
                ),
            }
        )

    hits.sort(key=lambda h: h["loss_pct"])
    return hits
