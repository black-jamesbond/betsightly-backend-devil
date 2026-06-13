"""
BetSightly Telegram Bot

Features:
- Parse single or multiple betting codes from one message
- Status updates via reply ("Won" / "Lost")
- /stats <punter> — punter performance
- /leaderboard — top punters
- /today — all codes posted today
- /verify <punter> — mark punter as verified (admin)
- /delete <code> — remove a betting code (admin)
- Duplicate code warnings
"""

import os
import re
import logging
from datetime import datetime
from typing import Dict, Any, List

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Import database module
try:
    from telegram_db import (
        SessionLocal, init_db,
        get_or_create_punter, get_or_create_bookmaker, save_betting_code,
        update_code_status, find_code_by_text,
        get_punter_stats, get_leaderboard, get_today_codes,
        verify_punter, delete_betting_code,
    )
    init_db()
    MOCK_MODE = False
    logger.info("Database connected. Running in REAL mode.")
except ImportError as e:
    logger.warning(f"Could not import database module: {str(e)}")
    logger.warning("Running in MOCK mode.")
    MOCK_MODE = True

# Config
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_GROUP_ID = os.getenv("TELEGRAM_GROUP_ID", "")

# Admin user IDs (Telegram numeric IDs) — add yours here
ADMIN_IDS = set(filter(None, os.getenv("TELEGRAM_ADMIN_IDS", "").split(",")))

# ── Regex patterns ──────────────────────────────────────────

PUNTER_PATTERN = r"Punter:\s*(.+)"
CODE_PATTERN = r"Code:\s*([A-Za-z0-9]+)"
ODDS_PATTERN = r"Odds:\s*([\d.]+)"
BOOKMAKER_PATTERN = r"(?:Bookmaker|Bookmarker):\s*([A-Za-z0-9\s]+)"

# Status update patterns (for replies)
STATUS_WON_PATTERN = r"^(?:won|win|w)\s*$"
STATUS_LOST_PATTERN = r"^(?:lost|lose|l)\s*$"


# ── Helpers ─────────────────────────────────────────────────


def is_admin(user_id: int) -> bool:
    """Check if a user is an admin."""
    return str(user_id) in ADMIN_IDS


def parse_multi_codes(text: str) -> List[Dict[str, Any]]:
    """
    Parse one or more betting code blocks from a single message.

    Supports formats like:
        Punter: King Jamz
        Code: ABC123
        Odds: 5.50
        Bookmaker: Sportybet

        Code: DEF456
        Odds: 3.20
        Bookmaker: Bet9ja

    The Punter line applies to all codes in the message.
    """
    # Get punter name (applies to entire message)
    punter_match = re.search(PUNTER_PATTERN, text, re.IGNORECASE)
    punter_name = punter_match.group(1).strip() if punter_match else None

    # Find all code blocks
    codes = re.findall(CODE_PATTERN, text, re.IGNORECASE)
    odds_list = re.findall(ODDS_PATTERN, text, re.IGNORECASE)
    bookmakers = re.findall(BOOKMAKER_PATTERN, text, re.IGNORECASE)

    if not codes:
        return []

    results = []
    for i, code in enumerate(codes):
        odds = float(odds_list[i]) if i < len(odds_list) else None
        bookmaker = bookmakers[i].strip() if i < len(bookmakers) else (
            bookmakers[0].strip() if bookmakers else None
        )

        results.append({
            "punter_name": punter_name,
            "bet_code": code,
            "odds": odds,
            "bookmaker": bookmaker,
            "event_date": datetime.now(),
        })

    return results


# ── Command Handlers ────────────────────────────────────────


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send welcome message with format instructions."""
    await update.message.reply_text(
        "Welcome to BetSightly Bot!\n\n"
        "Post betting codes in this format:\n\n"
        "Punter: King Jamz\n"
        "Code: ABC123\n"
        "Odds: 5.50\n"
        "Bookmaker: Sportybet\n\n"
        "You can post multiple codes in one message!\n"
        "Date is recorded automatically.\n\n"
        "Commands:\n"
        "/stats <name> - Punter stats\n"
        "/leaderboard - Top punters\n"
        "/today - Today's codes\n"
        "/help - Show this message"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show help text."""
    help_text = (
        "Post betting codes:\n\n"
        "Punter: King Jamz\n"
        "Code: ABC123\n"
        "Odds: 5.50\n"
        "Bookmaker: Sportybet\n\n"
        "Multiple codes in one message:\n\n"
        "Punter: King Jamz\n"
        "Code: ABC123\n"
        "Odds: 5.50\n"
        "Bookmaker: Sportybet\n\n"
        "Code: DEF456\n"
        "Odds: 3.20\n"
        "Bookmaker: Bet9ja\n\n"
        "Update status: Reply to any message containing a code with Won or Lost\n\n"
        "Commands:\n"
        "/stats <name> - Punter performance\n"
        "/leaderboard - Top 5 punters\n"
        "/today - All codes posted today\n"
        "/verify <name> - Verify a punter (admin)\n"
        "/delete <code> - Delete a code (admin)"
    )
    await update.message.reply_text(help_text)


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show stats for a punter: /stats King Jamz"""
    if not context.args:
        await update.message.reply_text("Usage: /stats <punter name>\nExample: /stats King Jamz")
        return

    if MOCK_MODE:
        await update.message.reply_text("Bot is in mock mode — no database available.")
        return

    punter_name = " ".join(context.args)
    db = SessionLocal()
    try:
        stats = get_punter_stats(db, punter_name)
        if not stats:
            await update.message.reply_text(f"Punter '{punter_name}' not found.")
            return

        verified = " [Verified]" if stats["verified"] else ""
        streak_emoji = ""
        if stats["streak"] >= 2:
            streak_emoji = " (on fire!)" if stats["streak_type"] == "won" else " (cold streak)"

        text = (
            f"--- {stats['name']}{verified} ---\n\n"
            f"Total Codes: {stats['total_codes']}\n"
            f"Won: {stats['won']}\n"
            f"Lost: {stats['lost']}\n"
            f"Pending: {stats['pending']}\n"
            f"Success Rate: {stats['success_rate']}%\n"
            f"Current Streak: {stats['streak']}{stats['streak_type'][0].upper() if stats['streak_type'] != 'none' else ''}{streak_emoji}"
        )
        await update.message.reply_text(text)
    finally:
        db.close()


async def leaderboard_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show top 5 punters."""
    if MOCK_MODE:
        await update.message.reply_text("Bot is in mock mode — no database available.")
        return

    db = SessionLocal()
    try:
        leaders = get_leaderboard(db, limit=5)
        if not leaders:
            await update.message.reply_text("No punters yet. Start posting codes!")
            return

        lines = ["--- Leaderboard ---\n"]
        medals = ["1st", "2nd", "3rd", "4th", "5th"]
        for i, p in enumerate(leaders):
            verified = " [V]" if p["verified"] else ""
            lines.append(
                f"{medals[i]}: {p['name']}{verified}\n"
                f"    {p['success_rate']}% | {p['won']}W-{p['lost']}L | {p['total_codes']} codes"
            )

        await update.message.reply_text("\n".join(lines))
    finally:
        db.close()


async def today_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show all codes posted today."""
    if MOCK_MODE:
        await update.message.reply_text("Bot is in mock mode — no database available.")
        return

    db = SessionLocal()
    try:
        codes = get_today_codes(db)
        if not codes:
            await update.message.reply_text("No codes posted today yet.")
            return

        lines = [f"--- Today's Codes ({len(codes)}) ---\n"]
        for c in codes:
            status_mark = {"won": "[W]", "lost": "[L]", "pending": "[?]"}.get(c["status"], "[?]")
            time_str = c["created_at"].strftime("%H:%M") if c["created_at"] else ""
            lines.append(
                f"{status_mark} {c['code']} | {c['odds']}x | {c['bookmaker']}\n"
                f"    by {c['punter_name']} at {time_str}"
            )

        await update.message.reply_text("\n".join(lines))
    finally:
        db.close()


async def verify_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Admin: verify a punter. /verify King Jamz"""
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("Only admins can use this command.")
        return

    if not context.args:
        await update.message.reply_text("Usage: /verify <punter name>")
        return

    if MOCK_MODE:
        await update.message.reply_text("Bot is in mock mode.")
        return

    punter_name = " ".join(context.args)
    db = SessionLocal()
    try:
        punter = verify_punter(db, punter_name)
        if punter:
            await update.message.reply_text(f"{punter.name} is now verified!")
        else:
            await update.message.reply_text(f"Punter '{punter_name}' not found.")
    finally:
        db.close()


async def delete_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Admin: delete a betting code. /delete ABC123"""
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("Only admins can use this command.")
        return

    if not context.args:
        await update.message.reply_text("Usage: /delete <code>")
        return

    if MOCK_MODE:
        await update.message.reply_text("Bot is in mock mode.")
        return

    code_str = context.args[0].upper()
    db = SessionLocal()
    try:
        success = delete_betting_code(db, code_str)
        if success:
            await update.message.reply_text(f"Code {code_str} deleted.")
        else:
            await update.message.reply_text(f"Code '{code_str}' not found.")
    finally:
        db.close()


# ── World Cup Commands ──────────────────────────────────────


async def wctips_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show today's World Cup tips: /wctips or /wctips 2026-06-12"""
    try:
        import json
        from pathlib import Path

        data_dir = Path(__file__).parent / "worldcup" / "data"
        preds_path = data_dir / "wc_predictions.json"

        if not preds_path.exists():
            await update.message.reply_text("No World Cup predictions available yet.")
            return

        with open(preds_path) as f:
            predictions = json.load(f)

        # Filter by date
        from datetime import datetime as dt
        target_date = context.args[0] if context.args else None

        if not target_date:
            today = dt.now().strftime("%Y-%m-%d")
            future = sorted(set(
                p["commence_time"][:10] for p in predictions
                if p["commence_time"][:10] >= today
            ))
            target_date = future[0] if future else today

        day_preds = [p for p in predictions if p["commence_time"].startswith(target_date)]

        if not day_preds:
            await update.message.reply_text(f"No World Cup matches on {target_date}")
            return

        # Format nice date
        nice_date = dt.strptime(target_date, "%Y-%m-%d").strftime("%A, %d %b %Y")

        lines = [f"World Cup Tips - {nice_date}\n{len(day_preds)} matches\n"]

        for p in day_preds:
            time_str = dt.fromisoformat(p["commence_time"].replace("Z", "+00:00")).strftime("%H:%M")
            conf_pct = round(p["confidence"] * 100)

            lines.append(f"{time_str} | {p['home_team']} vs {p['away_team']}")
            lines.append(f"  Tip: {p['prediction']} ({conf_pct}%)")

            # Show top tips if available
            for tip in p.get("top_tips", [])[1:]:
                tip_pct = round(tip["confidence"] * 100)
                lines.append(f"  Alt: {tip['tip']} ({tip_pct}%)")
            lines.append("")

        await update.message.reply_text("\n".join(lines))

    except Exception as e:
        logger.error(f"Error in wctips: {e}", exc_info=True)
        await update.message.reply_text(f"Error loading WC tips: {str(e)}")


async def wcacca_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show World Cup accumulator picks: /wcacca"""
    try:
        import json
        from pathlib import Path

        data_dir = Path(__file__).parent / "worldcup" / "data"
        preds_path = data_dir / "wc_predictions.json"

        if not preds_path.exists():
            await update.message.reply_text("No World Cup predictions available yet.")
            return

        with open(preds_path) as f:
            predictions = json.load(f)

        from datetime import datetime as dt
        today = dt.now().strftime("%Y-%m-%d")
        future = sorted(set(
            p["commence_time"][:10] for p in predictions
            if p["commence_time"][:10] >= today
        ))
        target_date = future[0] if future else today
        day_preds = [p for p in predictions if p["commence_time"].startswith(target_date)]

        if not day_preds:
            await update.message.reply_text("No upcoming WC matches for accumulators.")
            return

        nice_date = dt.strptime(target_date, "%Y-%m-%d").strftime("%a %d %b")

        # Collect all tips sorted by confidence
        all_tips = []
        for p in day_preds:
            for tip in p.get("top_tips", []):
                all_tips.append({
                    "match": f"{p['home_team']} vs {p['away_team']}",
                    "tip": tip["tip"],
                    "conf": tip["confidence"],
                    "match_id": p["match_id"],
                })

        all_tips.sort(key=lambda x: x["conf"], reverse=True)

        # Build safe acca (top 3 from different matches)
        safe = []
        used = set()
        for t in all_tips:
            if t["match_id"] not in used and t["conf"] >= 0.55:
                safe.append(t)
                used.add(t["match_id"])
            if len(safe) >= 3:
                break

        lines = [f"WC Accumulator - {nice_date}\n"]

        if safe:
            lines.append("SAFE ACCA (3 picks):")
            for t in safe:
                lines.append(f"  {t['match']}")
                lines.append(f"  -> {t['tip']} ({round(t['conf']*100)}%)")
            lines.append("")

        # Bold acca (top 5)
        bold = []
        used2 = set()
        for t in all_tips:
            if t["match_id"] not in used2 and t["conf"] >= 0.40:
                bold.append(t)
                used2.add(t["match_id"])
            if len(bold) >= 5:
                break

        if bold:
            lines.append("BOLD ACCA (5 picks):")
            for t in bold:
                lines.append(f"  {t['match']}")
                lines.append(f"  -> {t['tip']} ({round(t['conf']*100)}%)")

        await update.message.reply_text("\n".join(lines))

    except Exception as e:
        logger.error(f"Error in wcacca: {e}", exc_info=True)
        await update.message.reply_text(f"Error: {str(e)}")


# ── Message Handlers ────────────────────────────────────────


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle all non-command text messages."""
    if not update.message or not update.message.text:
        return

    chat_id = update.effective_chat.id
    logger.info(f"Message from {update.effective_user.full_name} in chat {chat_id}: {update.message.text[:80]}")

    # Check if message is in the target group
    if TELEGRAM_GROUP_ID and str(chat_id) != TELEGRAM_GROUP_ID:
        return

    text = update.message.text.strip()

    # ── Check for status update reply ───────────────────
    if update.message.reply_to_message:
        await handle_status_reply(update, text)
        return

    # ── Try to parse betting codes ──────────────────────
    code_blocks = parse_multi_codes(text)

    if not code_blocks:
        # Not a betting code message — ignore silently
        return

    # Check required fields
    first = code_blocks[0]
    if not first["punter_name"]:
        await update.message.reply_text(
            "Missing: Punter\n\n"
            "Format:\n"
            "Punter: King Jamz\n"
            "Code: ABC123\n"
            "Odds: 5.50\n"
            "Bookmaker: Sportybet"
        )
        return

    # Check each code block for required fields
    incomplete = []
    for i, block in enumerate(code_blocks):
        missing = []
        if not block["bet_code"]:
            missing.append("Code")
        if not block["odds"]:
            missing.append("Odds")
        if not block["bookmaker"]:
            missing.append("Bookmaker")
        if missing:
            incomplete.append(f"Code #{i+1}: missing {', '.join(missing)}")

    if incomplete:
        await update.message.reply_text(
            "Missing fields:\n" + "\n".join(incomplete) + "\n\n"
            "Each code needs: Code, Odds, Bookmaker"
        )
        return

    if MOCK_MODE:
        lines = [f"Received {len(code_blocks)} code(s) (MOCK MODE):"]
        for b in code_blocks:
            lines.append(f"  {b['bet_code']} | {b['odds']}x | {b['bookmaker']}")
        await update.message.reply_text("\n".join(lines))
        return

    # ── Save to database ────────────────────────────────
    db = SessionLocal()
    try:
        punter_name = first["punter_name"]
        punter_username = update.effective_user.username

        punter = get_or_create_punter(
            db, name=punter_name,
            telegram_username=str(update.effective_user.id),
            nickname=punter_username
        )
        if not punter:
            await update.message.reply_text("Failed to save punter. Please try again.")
            return

        saved = []
        duplicates = []

        for block in code_blocks:
            bookmaker = get_or_create_bookmaker(db, name=block["bookmaker"])
            if not bookmaker:
                await update.message.reply_text(f"Failed to save bookmaker: {block['bookmaker']}")
                continue

            result, is_dup = save_betting_code(
                db, code=block["bet_code"],
                punter_id=punter.id,
                bookmaker_id=bookmaker.id,
                odds=block["odds"],
                event_date=block["event_date"],
                notes=f"From Telegram: {punter_name}"
            )

            if result and is_dup:
                duplicates.append(block["bet_code"])
            elif result:
                saved.append(block)

        # Build response
        lines = []
        if saved:
            lines.append(f"Saved {len(saved)} code(s) for {punter_name}:\n")
            for s in saved:
                lines.append(f"  {s['bet_code']} | {s['odds']}x | {s['bookmaker']}")
            lines.append(f"\nDate: {datetime.now().strftime('%d %b %Y')}")

        if duplicates:
            lines.append(f"\nDuplicate(s) skipped: {', '.join(duplicates)}")
            lines.append("(These codes already exist in the system)")

        if not saved and not duplicates:
            lines.append("Failed to save codes. Please try again.")

        await update.message.reply_text("\n".join(lines))

    except Exception as e:
        logger.error(f"Error saving betting info: {str(e)}", exc_info=True)
        await update.message.reply_text(f"Error: {str(e)}\nPlease try again.")
    finally:
        db.close()


async def handle_status_reply(update: Update, text: str) -> None:
    """
    Handle a reply to a message — check if it's a status update.
    Reply "Won" or "Lost" to a message containing a code to update it.
    """
    if MOCK_MODE:
        return

    # Determine status from reply text
    text_lower = text.strip().lower()
    if re.match(STATUS_WON_PATTERN, text_lower):
        new_status = "won"
    elif re.match(STATUS_LOST_PATTERN, text_lower):
        new_status = "lost"
    else:
        return  # Not a status update

    # Extract code from the original message
    original_text = update.message.reply_to_message.text or ""
    code_match = re.search(CODE_PATTERN, original_text, re.IGNORECASE)

    if not code_match:
        await update.message.reply_text(
            "Could not find a betting code in the original message.\n"
            "Reply to a message that contains 'Code: ...' with Won or Lost."
        )
        return

    code_str = code_match.group(1)

    db = SessionLocal()
    try:
        code = find_code_by_text(db, code_str)
        if not code:
            await update.message.reply_text(f"Code '{code_str}' not found in the database.")
            return

        updated = update_code_status(db, code_str, new_status)
        if updated:
            # Get punter name for the notification
            from punter import Punter
            punter = db.query(Punter).get(updated.punter_id)
            punter_name = punter.name if punter else "Unknown"

            status_emoji = "W" if new_status == "won" else "L"
            await update.message.reply_text(
                f"[{status_emoji}] Code {code_str} marked as {new_status.upper()}!\n"
                f"Punter: {punter_name}\n"
                f"Odds: {updated.odds}x"
            )
        else:
            await update.message.reply_text(f"Failed to update code {code_str}.")
    finally:
        db.close()


# ── Main ────────────────────────────────────────────────────


def main() -> None:
    """Start the bot."""
    if not TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN not set.")
        return

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Commands
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(CommandHandler("leaderboard", leaderboard_command))
    application.add_handler(CommandHandler("today", today_command))
    application.add_handler(CommandHandler("verify", verify_command))
    application.add_handler(CommandHandler("delete", delete_command))
    application.add_handler(CommandHandler("wctips", wctips_command))
    application.add_handler(CommandHandler("wcacca", wcacca_command))

    # All text messages (codes + status replies)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("BetSightly Bot starting...")
    application.run_polling()


if __name__ == "__main__":
    main()
