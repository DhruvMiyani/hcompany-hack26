"""Browser task prompts — the exact instructions the H Company agent receives."""

from agent.tasks import execute_bet_task, add_funds_task, check_results_task


CREDS = ("https://demo.kalshi.co", "team@example.com", "hunter2")


def test_execute_searches_by_plain_title_not_ticker():
    task = execute_bet_task(
        *CREDS,
        market="[FRA vs ESP] France vs Spain Winner?",
        direction="Yes", amount=1.0,
        ticker="KXWCGAME-26JUL14FRAESP-TIE",
        outcome="Reg Time: Tie",
    )
    # search step uses the title with the short-ID prefix stripped
    assert 'search bar: "France vs Spain Winner?"' in task
    assert 'search bar: "[FRA' not in task
    assert "search by title, NOT by ticker" in task


def test_execute_names_the_exact_outcome_row():
    task = execute_bet_task(
        *CREDS, market="France vs Spain Winner?", direction="Yes",
        amount=1.0, ticker="KXWCGAME-26JUL14FRAESP-TIE", outcome="Reg Time: Tie",
    )
    assert 'outcome row "Reg Time: Tie"' in task
    assert 'Select the Yes side for outcome "Reg Time: Tie"' in task


def test_execute_without_outcome_omits_outcome_step():
    task = execute_bet_task(
        *CREDS, market="Both teams to score?", direction="No",
        amount=2.5, ticker="KXWCBTTS-26JUL14FRAESP-BTTS",
    )
    assert "outcome row" not in task
    assert "Select the No side" in task
    assert "$2.50" in task


def test_execute_pins_the_exact_bet():
    task = execute_bet_task(
        *CREDS, market="France vs Spain Winner?", direction="Yes",
        amount=5.0, ticker="KXWCGAME-26JUL14FRAESP-TIE",
    )
    assert "do not re-evaluate" in task
    assert "KXWCGAME-26JUL14FRAESP-TIE" in task
    assert "STATUS:" in task and "FILLED_PRICE:" in task and "ORDER_ID:" in task


def test_all_tasks_embed_credentials():
    for task in (
        execute_bet_task(*CREDS, market="X?", direction="Yes", amount=1.0),
        add_funds_task(*CREDS, amount=100.0),
        check_results_task(*CREDS),
    ):
        assert "team@example.com" in task
        assert "hunter2" in task
        assert "https://demo.kalshi.co" in task


def test_add_funds_mentions_amount():
    assert "$100.00" in add_funds_task(*CREDS, amount=100.0)
    assert "NEW_BALANCE" in add_funds_task(*CREDS)
