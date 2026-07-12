# Data trends — 1051 settled WC markets (train split)

## Calibration by price bucket

| Price | n | Implied | Actual YES | Edge |
|---|---|---|---|---|
| 00-10c | 215 | 0.05 | 0.05 | -0.00 |
| 10-20c | 161 | 0.14 | 0.16 | +0.02 |
| 20-30c | 152 | 0.24 | 0.20 | -0.05 |
| 30-40c | 121 | 0.35 | 0.31 | -0.03 |
| 40-50c | 143 | 0.44 | 0.46 | +0.02 |
| 50-60c | 87 | 0.55 | 0.62 | +0.07 |
| 60-70c | 55 | 0.65 | 0.67 | +0.03 |
| 70-80c | 48 | 0.74 | 0.83 | +0.09 |
| 80-90c | 29 | 0.84 | 0.79 | -0.04 |
| 90-100c | 40 | 0.93 | 0.95 | +0.02 |

## Favorite accuracy by category

| Category | n | Favorite accuracy |
|---|---|---|
| spread | 259 | 86% |
| total_goals | 235 | 83% |
| match_winner | 222 | 81% |
| first_half_winner | 220 | 69% |
| both_teams_score | 75 | 53% |
| advance | 40 | 75% |

## Momentum (final 24h price move)

- **rising**: n=58, implied 0.56, actual 0.55, edge -0.01
- **falling**: n=19, implied 0.34, actual 0.32, edge -0.02

## Liquidity

- deep (OI>100k): favorite accuracy 76% (n=479)
- mid (10k-100k): favorite accuracy 80% (n=433)
- thin (<=10k): favorite accuracy 78% (n=139)

## Derived strategy rules

1. Markets priced 50-60c settle YES +7% more often than the price implies — prefer YES there (n=87).
2. Markets priced 70-80c settle YES +9% more often than the price implies — prefer YES there (n=48).
3. Favorites are most reliable in spread (86% accurate, n=259).
4. Favorites are unreliable in both_teams_score (53%, n=75) — demand a bigger edge or skip.
