"""Prompt templates for browser agent tasks."""


# ── Phase 1: Scrape ──────────────────────────────────────────────────────────

def scrape_markets_task(url: str, email: str, password: str) -> str:
    return f"""
Go to {url} and log in with these credentials:
  Email: {email}
  Password: {password}

After logging in, navigate to the Markets section. Find ALL open FIFA or soccer prediction markets.

For each market you find, output one line in EXACTLY this format:
MARKET | YES_PRICE | NO_PRICE | VOLUME | CLOSES

Where:
  MARKET     = full market title (e.g. "Will Brazil win vs France?")
  YES_PRICE  = current Yes price as a decimal (e.g. 0.62)
  NO_PRICE   = current No price as a decimal (e.g. 0.38)
  VOLUME     = total dollar volume traded (e.g. 1420.50), or UNKNOWN
  CLOSES     = closing date/time (e.g. 2026-07-12 18:00 UTC), or UNKNOWN

List every open FIFA/soccer market you can find. Do not filter or skip any.
After the list, write: DONE

If you find no FIFA markets, write:
NO_MARKETS
DONE
"""


# ── Phase 3: Execute ─────────────────────────────────────────────────────────

def execute_bet_task(
    url: str,
    email: str,
    password: str,
    market: str,
    direction: str,
    amount: float,
    ticker: str = "",
    outcome: str = "",
) -> str:
    # Kalshi's search bar matches market titles, not raw tickers — strip the
    # short-ID prefix (e.g. "[FRA vs ESP] ") and search by the plain title.
    # Multi-outcome markets (e.g. "France vs Spain Winner?") are ONE page with
    # several outcome rows; `outcome` says which row to bet on (e.g. "Reg Time: Tie").
    import re
    search_title = re.sub(r"^\[[^\]]*\]\s*", "", market).strip()[:60]
    outcome_step = (
        f'3. On the market page, find the outcome row "{outcome}" '
        f"(the market has multiple outcomes — pick exactly this one)\n"
        if outcome else ""
    )
    n = 4 if outcome else 3
    return f"""
Go to {url} and log in with these credentials:
  Email: {email}
  Password: {password}

Place EXACTLY this bet — do not re-evaluate or choose a different market:

  Market ticker : {ticker or "N/A"}
  Market title  : {market[:80]}
  Outcome       : {outcome or "N/A"}
  Direction     : {direction}
  Amount        : ${amount:.2f}

Steps (execute quickly):
1. After login, type the market TITLE into the search bar: "{search_title}"
   (search by title, NOT by ticker — the ticker is only to confirm you found
   the right market once you're on its page)
2. Click into that specific market
{outcome_step}{n}. Select the {direction} side{f' for outcome "{outcome}"' if outcome else ""}
{n+1}. Enter ${amount:.2f} in the amount field
{n+2}. Click Confirm / Buy to submit the order

Report back with these exact labels:
STATUS: [Bet placed successfully / Error: <description>]
FILLED_PRICE: [actual fill price as decimal, e.g. 0.58]
ORDER_ID: [order ID from confirmation screen, or N/A]
"""


# ── Check results ─────────────────────────────────────────────────────────────

def check_results_task(url: str, email: str, password: str) -> str:
    return f"""
Go to {url} and log in with these credentials:
  Email: {email}
  Password: {password}

Navigate to your Portfolio page. Find all FIFA or soccer-related prediction market positions.

For EACH position found, output one line:
BET: [market title] | [Yes/No] | [$X wagered] | STATUS: [Open/Won/Lost] | P&L: [$X.XX or Pending]

After listing all bets, write:
SUMMARY: [total bets] bets, [total won] won, [total lost] lost, [net P&L] net
"""


# ── Add funds ───────────────────────────────────────────────────────────────

def add_funds_task(url: str, email: str, password: str, amount: float = 100.0) -> str:
    return f"""
Go to {url} and log in with these credentials:
  Email: {email}
  Password: {password}

This is a DEMO / paper-trading account. Add or reset the demo balance so it has
at least ${amount:.2f} of play money available to trade with.

Steps (try in order until the balance increases):
1. Open the account / wallet / settings menu (often the avatar or balance in the top-right).
2. Look for "Add demo funds", "Reset balance", "Deposit", "Add funds", or a
   similar play-money control.
3. If prompted for an amount, enter {amount:.0f}.
4. Confirm the action.
5. Return to the portfolio and read the new available cash balance.

Report back with these exact labels:
STATUS: [Funds added successfully / Error: <description>]
NEW_BALANCE: [available cash as a number, e.g. 100.00]
"""


# ── Reflection ────────────────────────────────────────────────────────────────

REFLECTION_SYSTEM = """You are a sports betting strategy analyst specialising in prediction markets.
Your job is to analyse past bet outcomes and extract actionable lessons to improve future performance.
Be data-driven, specific, and honest about what the results reveal.
Focus on patterns that are repeatable, not one-off luck."""


def reflection_prompt(bet_history: str, current_strategy: str, existing_lessons: str) -> str:
    return f"""
Analyse these past FIFA betting outcomes on Kalshi:

=== RESOLVED BETS ===
{bet_history}

=== CURRENT STRATEGY ===
{current_strategy}

=== EXISTING LESSONS ===
{existing_lessons}

Based on this data:
1. What patterns distinguish winning bets from losing ones?
2. Are there specific market types, price ranges, or conditions where performance is better?
3. What 2-3 new lessons should we add or refine?
4. Should we adjust bet sizing?

Return a JSON object with this structure:
{{
  "new_lessons": [
    {{"lesson": "...", "confidence": 0.8, "applies_when": "..."}}
  ],
  "updated_strategy_rules": ["rule 1", "rule 2", ...],
  "bet_size_recommendation": "increase|decrease|maintain",
  "summary": "One paragraph summary of what we learned"
}}
"""
