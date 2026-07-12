#!/usr/bin/env python3
"""
Find trends in the REAL settled-market data and turn them into strategy rules.

  python analyze_trends.py [--save]     (--save stores rules into agent memory)

Looks for the classic prediction-market edges on the TRAIN split only:
  - calibration by price bucket (favorite-longshot bias)
  - accuracy of the market favorite by category
  - price momentum over the final 24h pre-match
  - liquidity (open interest) effects

Report → data/trends_report.md, rules → data/trend_rules.json
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

from agent.dataset import load_dataset

REPORT_PATH = Path("data") / "trends_report.md"
RULES_PATH = Path("data") / "trend_rules.json"


def _bucket(p: float) -> str:
    lo = int(p * 10) * 10
    return f"{lo:02d}-{lo + 10:02d}c"


def calibration_table(examples) -> list[dict]:
    buckets = defaultdict(list)
    for ex in examples:
        buckets[_bucket(ex["yes_price_4h"])].append(ex)
    rows = []
    for b in sorted(buckets):
        exs = buckets[b]
        implied = sum(e["yes_price_4h"] for e in exs) / len(exs)
        actual = sum(e["result"] for e in exs) / len(exs)
        rows.append({"bucket": b, "n": len(exs),
                     "implied": round(implied, 3), "actual": round(actual, 3),
                     "edge": round(actual - implied, 3)})
    return rows


def favorite_accuracy_by_category(examples) -> list[dict]:
    cats = defaultdict(list)
    for ex in examples:
        cats[ex["category"]].append(ex)
    rows = []
    for cat, exs in sorted(cats.items(), key=lambda kv: -len(kv[1])):
        fav_correct = sum(
            1 for e in exs
            if (e["yes_price_4h"] > 0.5) == bool(e["result"])
        )
        rows.append({"category": cat, "n": len(exs),
                     "favorite_accuracy": round(fav_correct / len(exs), 3)})
    return rows


def momentum_signal(examples, threshold: float = 0.03) -> dict:
    rising = [e for e in examples if e["momentum"] > threshold]
    falling = [e for e in examples if e["momentum"] < -threshold]

    def yes_rate_vs_implied(exs):
        if not exs:
            return None
        implied = sum(e["yes_price_4h"] for e in exs) / len(exs)
        actual = sum(e["result"] for e in exs) / len(exs)
        return {"n": len(exs), "implied": round(implied, 3),
                "actual": round(actual, 3), "edge": round(actual - implied, 3)}

    return {"rising": yes_rate_vs_implied(rising),
            "falling": yes_rate_vs_implied(falling)}


def liquidity_effect(examples) -> list[dict]:
    tiers = [("deep (OI>100k)", lambda v: v > 100_000),
             ("mid (10k-100k)", lambda v: 10_000 < v <= 100_000),
             ("thin (<=10k)", lambda v: v <= 10_000)]
    rows = []
    for label, pred in tiers:
        exs = [e for e in examples if pred(e.get("open_interest") or 0)]
        if not exs:
            continue
        fav = sum(1 for e in exs if (e["yes_price_4h"] > 0.5) == bool(e["result"]))
        rows.append({"tier": label, "n": len(exs),
                     "favorite_accuracy": round(fav / len(exs), 3)})
    return rows


def derive_rules(calib, by_cat, momentum, liquidity) -> list[str]:
    """Turn statistically visible edges into plain-English strategy rules."""
    rules = []
    for row in calib:
        if row["n"] >= 15 and row["edge"] >= 0.05:
            rules.append(f"Markets priced {row['bucket']} settle YES "
                         f"{row['edge']:+.0%} more often than the price implies "
                         f"— prefer YES there (n={row['n']}).")
        elif row["n"] >= 15 and row["edge"] <= -0.05:
            rules.append(f"Markets priced {row['bucket']} are overpriced by "
                         f"{-row['edge']:.0%} — prefer NO there (n={row['n']}).")
    best = [r for r in by_cat if r["n"] >= 20]
    if best:
        top = max(best, key=lambda r: r["favorite_accuracy"])
        worst = min(best, key=lambda r: r["favorite_accuracy"])
        rules.append(f"Favorites are most reliable in {top['category']} "
                     f"({top['favorite_accuracy']:.0%} accurate, n={top['n']}).")
        if worst["favorite_accuracy"] < 0.55:
            rules.append(f"Favorites are unreliable in {worst['category']} "
                         f"({worst['favorite_accuracy']:.0%}, n={worst['n']}) "
                         f"— demand a bigger edge or skip.")
    for side, stats in momentum.items():
        if stats and stats["n"] >= 15 and abs(stats["edge"]) >= 0.04:
            direction = "YES" if stats["edge"] > 0 else "NO"
            rules.append(f"Price {side} over the final 24h → lean {direction} "
                         f"(edge {stats['edge']:+.0%}, n={stats['n']}).")
    if liquidity:
        deep = next((r for r in liquidity if "deep" in r["tier"]), None)
        thin = next((r for r in liquidity if "thin" in r["tier"]), None)
        if deep and thin and deep["favorite_accuracy"] - thin["favorite_accuracy"] > 0.05:
            rules.append("Deep markets price more accurately than thin ones — "
                         "trust prices in liquid markets, look for edge in thin ones.")
    return rules


def render_report(n, calib, by_cat, momentum, liquidity, rules) -> str:
    md = [f"# Data trends — {n} settled WC markets (train split)\n"]
    md.append("## Calibration by price bucket\n")
    md.append("| Price | n | Implied | Actual YES | Edge |")
    md.append("|---|---|---|---|---|")
    for r in calib:
        md.append(f"| {r['bucket']} | {r['n']} | {r['implied']:.2f} "
                  f"| {r['actual']:.2f} | {r['edge']:+.2f} |")
    md.append("\n## Favorite accuracy by category\n")
    md.append("| Category | n | Favorite accuracy |")
    md.append("|---|---|---|")
    for r in by_cat:
        md.append(f"| {r['category']} | {r['n']} | {r['favorite_accuracy']:.0%} |")
    md.append("\n## Momentum (final 24h price move)\n")
    for side, s in momentum.items():
        md.append(f"- **{side}**: " + (f"n={s['n']}, implied {s['implied']:.2f}, "
                  f"actual {s['actual']:.2f}, edge {s['edge']:+.2f}" if s else "no data"))
    md.append("\n## Liquidity\n")
    for r in liquidity:
        md.append(f"- {r['tier']}: favorite accuracy {r['favorite_accuracy']:.0%} "
                  f"(n={r['n']})")
    md.append("\n## Derived strategy rules\n")
    md.extend(f"{i + 1}. {r}" for i, r in enumerate(rules)) if rules else md.append("_No edges cleared the significance bar._")
    return "\n".join(md) + "\n"


def run(save_to_memory: bool = False, log=print) -> list[str]:
    train, _ = load_dataset()
    if not train:
        raise SystemExit("No train set — run `python build_dataset.py` first.")
    calib = calibration_table(train)
    by_cat = favorite_accuracy_by_category(train)
    momentum = momentum_signal(train)
    liquidity = liquidity_effect(train)
    rules = derive_rules(calib, by_cat, momentum, liquidity)

    REPORT_PATH.write_text(render_report(len(train), calib, by_cat,
                                         momentum, liquidity, rules))
    RULES_PATH.write_text(json.dumps(rules, indent=2))
    log(f"Report → {REPORT_PATH}")
    log(f"Rules  → {RULES_PATH}\n")
    for r in rules:
        log(f"  • {r}")

    if save_to_memory and rules:
        from agent import memory
        memory.init_db()
        base = memory.get_latest_strategy()
        merged = base + [r for r in rules if r not in base]
        memory.save_strategy(merged)
        log(f"\nSaved {len(rules)} data-driven rules into live strategy "
            f"({len(merged)} total).")
    return rules


if __name__ == "__main__":
    run(save_to_memory="--save" in sys.argv)
