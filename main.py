#!/usr/bin/env python3
"""
Self-improving Kalshi FIFA betting agent — powered by H Company browser automation.

  python main.py bet       — API fetch → GRPO decide → Holo validate → browser execute
  python main.py simulate  — generate synthetic data + GRPO cold-start training
  python main.py train     — retrain GRPO on real resolved bet outcomes
  python main.py check     — check portfolio + reflection + GRPO retrain
  python main.py stats     — performance history and current strategy
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

load_dotenv(Path(__file__).parent / ".env")

from agent import memory, reflection, runner

console = Console()


def cmd_bet():
    if not os.getenv("HAI_API_KEY"):
        console.print("[red]HAI_API_KEY not set in .env[/red]")
        sys.exit(1)
    if not os.getenv("KALSHI_EMAIL"):
        console.print("[red]KALSHI_EMAIL not set in .env[/red]")
        sys.exit(1)

    max_amount = float(os.getenv("MAX_BET_AMOUNT", "5.00"))

    memory.init_db()
    strategy = memory.get_latest_strategy()
    lessons = memory.get_active_lessons()

    from agent.grpo_model import get_model as get_grpo
    grpo_status = "fine-tuned ✓" if get_grpo().is_trained else "base (run simulate first)"
    console.print(Panel(
        f"[bold cyan]Self-improving two-model pipeline[/bold cyan]\n\n"
        f"  Phase 1   Kalshi REST API  →  pure KXWC WC markets\n"
        f"  Phase 2A  GRPO Model       →  Qwen2.5-1.5B [{grpo_status}]\n"
        f"  Phase 2B  Holo Validator   →  holo3-1-35b-a3b sanity check\n"
        f"  Phase 3   H Company        →  h/web-surfer-flash executes\n\n"
        f"Target: [green]{os.getenv('KALSHI_URL')}[/green]  |  "
        f"Max bet: [green]${max_amount:.2f}[/green]  |  "
        f"Lessons: {len(lessons)}",
        title="Kalshi FIFA Agent",
    ))

    result = runner.run_full_bet_cycle(
        strategy_rules=strategy,
        lessons=lessons,
        max_amount=max_amount,
    )

    decision = result["decision"]

    if result["skipped"]:
        console.print(Panel(
            f"[yellow]Skipped — no suitable market found.[/yellow]\n{result.get('skip_reason', '')}",
            title="Phase 2 — Decision",
        ))
        if result.get("raw_markets"):
            console.print(f"\n[dim]Raw market data from browser:[/dim]\n{result['raw_markets']}")
        return

    # Decision summary
    d_table = Table(title="Phase 2 — Holo Model Decision", show_header=False)
    d_table.add_column("Field", style="cyan")
    d_table.add_column("Value")
    d_table.add_row("Market", decision["market"])
    d_table.add_row("Direction", f"[green]{decision['direction']}[/green]")
    d_table.add_row("Amount", f"${decision['amount']:.2f}")
    d_table.add_row("Confidence", f"{decision['confidence']:.0%}")
    d_table.add_row("Reasoning", decision["reasoning"])
    console.print(d_table)

    # Execution result
    exec_info = result.get("execution") or {}
    if exec_info.get("success"):
        bet_id = memory.store_bet(
            session_id=result["session_ids"]["execute"],
            market=decision["market"],
            direction=decision["direction"],
            amount=decision["amount"],
            odds=exec_info.get("filled_price"),
            raw_answer=exec_info.get("answer", ""),
        )
        console.print(Panel(
            f"[green]Bet placed successfully![/green]\n\n"
            f"  Filled price : {exec_info.get('filled_price') or 'N/A'}\n"
            f"  Order ID     : {exec_info.get('order_id') or 'N/A'}\n"
            f"  Local bet ID : {bet_id}",
            title="Phase 3 — Execution",
        ))
    else:
        console.print(Panel(
            f"[red]Execution failed.[/red]\n\n{exec_info.get('answer', '')}",
            title="Phase 3 — Execution",
        ))


def cmd_check():
    if not os.getenv("HAI_API_KEY"):
        console.print("[red]HAI_API_KEY not set in .env[/red]")
        sys.exit(1)

    memory.init_db()
    console.print(Panel("[bold]Checking Kalshi demo portfolio...[/bold]", title="Portfolio Check"))

    result = runner.run_check_results()
    answer = result.get("answer", "")
    console.print(f"\n[dim]Agent answer:[/dim]\n{answer}\n")

    browser_bets = result.get("bets", [])
    pending_db = memory.get_pending_bets()

    updated = 0
    for b_db in pending_db:
        for b_live in browser_bets:
            if _markets_match(b_db["market"], b_live["market"]):
                if b_live["status"] in ("won", "lost"):
                    memory.update_bet_outcome(b_db["id"], b_live["status"], b_live.get("profit_loss") or 0.0)
                    updated += 1

    console.print(f"[green]Updated {updated} bet(s) with outcomes.[/green]")

    resolved = memory.get_resolved_bets(limit=20)
    if resolved:
        console.print("\n[bold cyan]Running self-improvement reflection...[/bold cyan]")
        strategy = memory.get_latest_strategy()
        lessons = memory.get_active_lessons()

        with console.status("Calling Holo model for reflection..."):
            output = reflection.reflect_on_outcomes(resolved, strategy, lessons)

        if output:
            for l in output.get("new_lessons", []):
                memory.store_lesson(l["lesson"], l.get("confidence", 0.5), l.get("applies_when", ""))
            if output.get("updated_strategy_rules"):
                memory.save_strategy(output["updated_strategy_rules"])

            console.print(Panel(
                f"Sizing: [bold]{output.get('bet_size_recommendation', 'maintain')}[/bold]  |  "
                f"New lessons: {len(output.get('new_lessons', []))}\n\n"
                f"{output.get('summary', '')}",
                title="Self-Improvement Result",
            ))
    else:
        console.print("[dim]No resolved bets yet — come back after markets close.[/dim]")


def cmd_stats():
    memory.init_db()
    perf = memory.get_performance_summary()
    strategy = memory.get_latest_strategy()
    lessons = memory.get_active_lessons()
    bets = memory.get_all_bets(limit=10)

    total = perf.get("total_bets") or 0
    wins = perf.get("wins") or 0
    losses = perf.get("losses") or 0
    wr = f"{wins / max(wins + losses, 1) * 100:.0f}%" if (wins + losses) > 0 else "N/A"
    pl = perf.get("total_profit_loss") or 0

    console.print(Panel(
        f"Bets: {total}  |  Wins: {wins}  |  Losses: {losses}  |  Win rate: {wr}\n"
        f"Wagered: ${perf.get('total_wagered') or 0:.2f}  |  Net P&L: ${pl:.2f}",
        title="Performance",
    ))

    console.print("\n[bold cyan]Current Strategy:[/bold cyan]")
    for i, r in enumerate(strategy, 1):
        console.print(f"  {i}. {r}")

    if lessons:
        console.print("\n[bold cyan]Learned Lessons:[/bold cyan]")
        for l in lessons[:10]:
            console.print(f"  [{l['confidence']:.0%}] {l['lesson']}")

    if bets:
        table = Table(title="\nRecent Bets")
        table.add_column("Date", style="dim")
        table.add_column("Market")
        table.add_column("Dir")
        table.add_column("$", justify="right")
        table.add_column("Status")
        table.add_column("P&L", justify="right")
        for b in bets:
            color = {"won": "green", "lost": "red", "pending": "yellow"}.get(b["status"], "white")
            pl_str = f"${b['profit_loss']:.2f}" if b.get("profit_loss") is not None else "-"
            table.add_row(
                b["created_at"][:10],
                b["market"][:50],
                b["direction"],
                f"${b['amount']:.2f}",
                f"[{color}]{b['status']}[/{color}]",
                pl_str,
            )
        console.print(table)


def _markets_match(a: str, b: str) -> bool:
    a_l, b_l = a.lower().strip(), b.lower().strip()
    if a_l == b_l:
        return True
    wa, wb = set(a_l.split()), set(b_l.split())
    return len(wa & wb) / max(len(wa | wb), 1) > 0.6


def cmd_simulate():
    """Generate synthetic GRPO training data + run initial fine-tuning."""
    memory.init_db()
    strategy = memory.get_latest_strategy()

    console.print(Panel(
        "[bold cyan]GRPO Cold Start[/bold cyan]\n\n"
        "  1. Fetch live WC markets from Kalshi API\n"
        "  2. Monte Carlo simulate 100 bet outcomes\n"
        "  3. Fine-tune Qwen2.5-0.5B with GRPO on simulated P&L rewards\n\n"
        "[dim]First run downloads ~1GB model — trains in minutes on M4[/dim]",
        title="GRPO Simulator",
    ))

    from agent.kalshi_api import get_open_wc_markets
    from agent.simulator import generate_trajectories
    from agent.grpo_model import get_model

    console.print("\n[cyan]Fetching live Kalshi WC markets...[/cyan]")
    markets = get_open_wc_markets()
    console.print(f"  {len(markets)} markets fetched")

    console.print("\n[cyan]Generating 100 synthetic trajectories...[/cyan]")
    trajectories = generate_trajectories(markets, strategy, n=100)
    wins = sum(1 for t in trajectories if t["won"])
    avg_reward = sum(t["reward"] for t in trajectories) / len(trajectories)
    console.print(f"  Simulated: {len(trajectories)} | Win rate: {wins/len(trajectories):.0%} | Avg reward: {avg_reward:.3f}")

    console.print("\n[cyan]Running GRPO fine-tuning on Qwen2.5-1.5B...[/cyan]")
    grpo = get_model()
    trained = grpo.train(trajectories)

    if trained:
        console.print(Panel(
            "[green]GRPO training complete![/green]\n\n"
            f"  Model: Qwen2.5-1.5B + LoRA adapter\n"
            f"  Trajectories: {len(trajectories)}\n"
            f"  Weights saved: data/grpo_weights/adapter/\n\n"
            "Run [bold]python main.py bet[/bold] — GRPO model is now Phase 2A",
            title="Training Complete",
        ))
    else:
        console.print("[yellow]Training skipped — check logs above.[/yellow]")


def cmd_train():
    """Retrain GRPO on actual resolved bet outcomes from SQLite."""
    memory.init_db()
    resolved = memory.get_resolved_bets(limit=50)

    if not resolved:
        console.print("[yellow]No resolved bets yet. Run 'check' after markets settle.[/yellow]")
        return

    from agent.grpo_model import get_model
    from agent.reward import compute_reward
    from agent.kalshi_api import get_open_wc_markets
    from agent.simulator import _build_prompt

    strategy = memory.get_latest_strategy()
    markets  = get_open_wc_markets()
    prompt   = _build_prompt(markets[:12], strategy)

    trajectories = []
    for bet in resolved:
        reward = compute_reward(
            decision={"direction": bet["direction"], "amount": bet["amount"],
                      "confidence": 0.5, "volume": None},
            outcome={"status": bet["status"], "profit_loss": bet.get("profit_loss", 0)},
        )
        trajectories.append({"prompt": prompt, "reward": reward})

    console.print(f"\n[cyan]Retraining GRPO on {len(trajectories)} real bet outcomes...[/cyan]")
    grpo = get_model()
    grpo.train(trajectories)


COMMANDS = {"bet": cmd_bet, "check": cmd_check, "stats": cmd_stats,
            "simulate": cmd_simulate, "train": cmd_train}

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "stats"
    if cmd not in COMMANDS:
        console.print(f"[red]Unknown command: {cmd}[/red]  Available: {', '.join(COMMANDS)}")
        sys.exit(1)
    COMMANDS[cmd]()
