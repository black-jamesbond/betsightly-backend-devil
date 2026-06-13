"""
Telegram Bot Database Connection

Uses the SAME database and models as the main API so that punters
created via Telegram appear in the frontend automatically.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from sqlalchemy import func, desc, case

from database import SessionLocal
from punter import Punter
from bookmaker import Bookmaker
from betting_code import BettingCode
from punter_prediction import PunterPrediction  # needed so SQLAlchemy resolves Punter.punter_predictions

logger = logging.getLogger(__name__)


def init_db():
    """Database is initialized by the main app — nothing to do here."""
    logger.info("Telegram DB: using main application database")
    return True


# ── Core CRUD ───────────────────────────────────────────────


def get_or_create_punter(db, name, telegram_username=None, nickname=None):
    """Get or create a punter in the MAIN punters table."""
    try:
        # Look up by name first (since Punter: field is the identifier now)
        punter = db.query(Punter).filter(
            func.lower(Punter.name) == name.lower()
        ).first()
        if punter:
            # Update telegram_username if we didn't have it
            if telegram_username and not punter.telegram_username:
                punter.telegram_username = telegram_username
                db.commit()
            return punter

        # Fall back to telegram username lookup
        if telegram_username:
            punter = db.query(Punter).filter(
                Punter.telegram_username == telegram_username
            ).first()
            if punter:
                return punter

        # Create new punter
        punter = Punter(
            name=name,
            nickname=nickname,
            telegram_username=telegram_username,
            country="Nigeria",
            specialty="betting_codes",
            verified=False,
        )

        db.add(punter)
        db.commit()
        db.refresh(punter)

        logger.info(f"Created punter: {name} (ID: {punter.id})")
        return punter

    except Exception as e:
        db.rollback()
        logger.error(f"Error getting or creating punter: {str(e)}")
        return None


def get_or_create_bookmaker(db, name):
    """Get or create a bookmaker."""
    try:
        bookmaker = db.query(Bookmaker).filter(
            func.lower(Bookmaker.name) == name.strip().lower()
        ).first()
        if bookmaker:
            return bookmaker

        bookmaker = Bookmaker(name=name.strip())
        db.add(bookmaker)
        db.commit()
        db.refresh(bookmaker)

        logger.info(f"Created bookmaker: {name} (ID: {bookmaker.id})")
        return bookmaker

    except Exception as e:
        db.rollback()
        logger.error(f"Error getting or creating bookmaker: {str(e)}")
        return None


def save_betting_code(db, code, punter_id, bookmaker_id=None, odds=None,
                      event_date=None, notes=None):
    """Save a betting code. Returns (betting_code, is_duplicate)."""
    try:
        existing = db.query(BettingCode).filter(BettingCode.code == code).first()
        if existing:
            logger.info(f"Betting code already exists: {code}")
            return existing, True  # duplicate

        betting_code = BettingCode(
            code=code,
            punter_id=punter_id,
            bookmaker_id=bookmaker_id,
            odds=odds,
            event_date=event_date,
            status="pending",
            confidence=8,
            featured=False,
            notes=notes,
        )

        db.add(betting_code)

        # Update punter popularity (total codes submitted)
        punter = db.query(Punter).get(punter_id)
        if punter:
            punter.popularity = (punter.popularity or 0) + 1
            punter.updated_at = datetime.now()

        db.commit()
        db.refresh(betting_code)

        logger.info(f"Saved betting code: {code} (ID: {betting_code.id})")
        return betting_code, False  # new

    except Exception as e:
        db.rollback()
        logger.error(f"Error saving betting code: {str(e)}")
        return None, False


# ── Status Updates ──────────────────────────────────────────


def update_code_status(db, code_str: str, new_status: str) -> Optional[BettingCode]:
    """Update a betting code's status and the punter's win/loss counts."""
    try:
        code = db.query(BettingCode).filter(BettingCode.code == code_str).first()
        if not code:
            return None

        old_status = code.status
        code.status = new_status
        code.updated_at = datetime.now()

        # Update punter stats
        punter = db.query(Punter).get(code.punter_id)
        if punter:
            # Undo old status
            if old_status == "won":
                punter.total_won = max((punter.total_won or 0) - 1, 0)
            elif old_status == "lost":
                punter.total_lost = max((punter.total_lost or 0) - 1, 0)

            # Apply new status
            if new_status == "won":
                punter.total_won = (punter.total_won or 0) + 1
            elif new_status == "lost":
                punter.total_lost = (punter.total_lost or 0) + 1

            # Recalculate success rate
            total = (punter.total_won or 0) + (punter.total_lost or 0)
            punter.success_rate = round(((punter.total_won or 0) / total) * 100, 1) if total > 0 else 0.0
            punter.updated_at = datetime.now()

        db.commit()
        db.refresh(code)
        return code

    except Exception as e:
        db.rollback()
        logger.error(f"Error updating code status: {str(e)}")
        return None


def find_code_by_text(db, code_str: str) -> Optional[BettingCode]:
    """Find a betting code by its code string."""
    return db.query(BettingCode).filter(BettingCode.code == code_str).first()


# ── Stats & Leaderboard ────────────────────────────────────


def get_punter_stats(db, punter_name: str) -> Optional[Dict[str, Any]]:
    """Get stats for a punter by name."""
    punter = db.query(Punter).filter(
        func.lower(Punter.name) == punter_name.strip().lower()
    ).first()
    if not punter:
        return None

    codes = db.query(BettingCode).filter(BettingCode.punter_id == punter.id).all()
    total = len(codes)
    won = sum(1 for c in codes if c.status == "won")
    lost = sum(1 for c in codes if c.status == "lost")
    pending = sum(1 for c in codes if c.status == "pending")

    # Calculate streak
    sorted_codes = sorted(
        [c for c in codes if c.status in ("won", "lost")],
        key=lambda c: c.created_at or datetime.min,
        reverse=True
    )
    streak = 0
    streak_type = None
    for c in sorted_codes:
        if streak_type is None:
            streak_type = c.status
            streak = 1
        elif c.status == streak_type:
            streak += 1
        else:
            break

    return {
        "name": punter.name,
        "verified": punter.verified,
        "total_codes": total,
        "won": won,
        "lost": lost,
        "pending": pending,
        "success_rate": round((won / (won + lost)) * 100, 1) if (won + lost) > 0 else 0.0,
        "streak": streak,
        "streak_type": streak_type or "none",
    }


def get_leaderboard(db, limit: int = 5) -> List[Dict[str, Any]]:
    """Get top punters by success rate (min 2 resolved codes)."""
    punters = db.query(Punter).all()
    results = []

    for p in punters:
        won = p.total_won or 0
        lost = p.total_lost or 0
        total_resolved = won + lost
        if total_resolved < 1:
            # Include punters with at least 1 code for early stage
            codes_count = db.query(func.count(BettingCode.id)).filter(
                BettingCode.punter_id == p.id
            ).scalar() or 0
            if codes_count == 0:
                continue
            results.append({
                "name": p.name,
                "verified": p.verified,
                "won": won,
                "lost": lost,
                "pending": codes_count - total_resolved,
                "success_rate": 0.0,
                "total_codes": codes_count,
            })
        else:
            rate = round((won / total_resolved) * 100, 1)
            codes_count = db.query(func.count(BettingCode.id)).filter(
                BettingCode.punter_id == p.id
            ).scalar() or 0
            results.append({
                "name": p.name,
                "verified": p.verified,
                "won": won,
                "lost": lost,
                "pending": codes_count - total_resolved,
                "success_rate": rate,
                "total_codes": codes_count,
            })

    # Sort by success rate desc, then total codes desc
    results.sort(key=lambda x: (x["success_rate"], x["total_codes"]), reverse=True)
    return results[:limit]


def get_today_codes(db) -> List[Dict[str, Any]]:
    """Get all codes posted today."""
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    codes = (
        db.query(BettingCode)
        .filter(BettingCode.created_at >= today_start)
        .order_by(desc(BettingCode.created_at))
        .all()
    )

    results = []
    for c in codes:
        punter = db.query(Punter).get(c.punter_id)
        bookmaker = db.query(Bookmaker).get(c.bookmaker_id) if c.bookmaker_id else None
        results.append({
            "code": c.code,
            "punter_name": punter.name if punter else "Unknown",
            "bookmaker": bookmaker.name if bookmaker else "Unknown",
            "odds": c.odds,
            "status": c.status,
            "created_at": c.created_at,
        })

    return results


# ── Admin ───────────────────────────────────────────────────


def verify_punter(db, punter_name: str) -> Optional[Punter]:
    """Mark a punter as verified."""
    try:
        punter = db.query(Punter).filter(
            func.lower(Punter.name) == punter_name.strip().lower()
        ).first()
        if not punter:
            return None

        punter.verified = True
        punter.updated_at = datetime.now()
        db.commit()
        db.refresh(punter)
        return punter

    except Exception as e:
        db.rollback()
        logger.error(f"Error verifying punter: {str(e)}")
        return None


def delete_betting_code(db, code_str: str) -> bool:
    """Delete a betting code and adjust punter stats."""
    try:
        code = db.query(BettingCode).filter(BettingCode.code == code_str).first()
        if not code:
            return False

        # Adjust punter stats
        punter = db.query(Punter).get(code.punter_id)
        if punter:
            punter.popularity = max((punter.popularity or 0) - 1, 0)
            if code.status == "won":
                punter.total_won = max((punter.total_won or 0) - 1, 0)
            elif code.status == "lost":
                punter.total_lost = max((punter.total_lost or 0) - 1, 0)
            # Recalculate
            total = (punter.total_won or 0) + (punter.total_lost or 0)
            punter.success_rate = round(((punter.total_won or 0) / total) * 100, 1) if total > 0 else 0.0
            punter.updated_at = datetime.now()

        db.delete(code)
        db.commit()
        return True

    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting betting code: {str(e)}")
        return False
