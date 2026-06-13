"""
API endpoints for fixtures.
"""

from typing import List, Optional
from datetime import datetime, date
from fastapi import APIRouter, Depends, HTTPException, Query

from sqlalchemy.orm import Session

from database import get_db
from fixture import Fixture
from prediction import Prediction
from services.fixture_service import FixtureService
from services.apifootball_service import APIFootballService

router = APIRouter()

@router.get("/")
def get_fixtures(
    db: Session = Depends(get_db),
    date: Optional[date] = None,
    league_id: Optional[int] = None
):
    """Get fixtures."""
    fixture_service = FixtureService(db)

    if date and league_id:
        # Get fixtures by date and league
        fixtures = fixture_service.get_fixtures_by_league(
            league_id=league_id,
            date=datetime.combine(date, datetime.min.time())
        )
    elif date:
        # Get fixtures by date
        fixtures = fixture_service.get_fixtures_by_date(
            date=datetime.combine(date, datetime.min.time())
        )
    elif league_id:
        # Get fixtures by league
        fixtures = fixture_service.get_fixtures_by_league(league_id=league_id)
    else:
        # Get today's fixtures
        fixtures = fixture_service.get_fixtures_by_date(
            date=datetime.now()
        )

    return [fixture.to_dict() for fixture in fixtures]

@router.get("/{fixture_id}")
def get_fixture(
    fixture_id: int,
    db: Session = Depends(get_db)
):
    """Get fixture by ID."""
    fixture_service = FixtureService(db)
    fixture = fixture_service.get_fixture_by_id(fixture_id)

    if not fixture:
        raise HTTPException(status_code=404, detail="Fixture not found")

    return fixture.to_dict()

@router.get("/{fixture_id}/prediction")
def get_fixture_prediction(
    fixture_id: int,
    db: Session = Depends(get_db)
):
    """Get prediction for fixture."""
    # Query prediction directly from database
    prediction = db.query(Prediction).filter(Prediction.fixture_id == fixture_id).first()

    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")

    return prediction.to_dict()


# APIFootball.com Testing Endpoints
@router.get("/apifootball/test")
def test_apifootball_connection():
    """Test APIFootball.com connection."""
    try:
        service = APIFootballService()
        is_connected = service.test_connection()

        return {
            "status": "success" if is_connected else "failed",
            "message": "APIFootball.com connection test",
            "connected": is_connected,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error testing APIFootball.com connection: {str(e)}",
            "connected": False,
            "timestamp": datetime.now().isoformat()
        }


@router.get("/apifootball/daily")
def get_apifootball_daily_fixtures(
    date: Optional[str] = Query(None, description="Date in YYYY-MM-DD format (defaults to today)")
):
    """Get daily fixtures from APIFootball.com for testing."""
    try:
        service = APIFootballService()
        fixtures = service.get_daily_fixtures(date)

        return {
            "status": "success",
            "message": f"Retrieved fixtures from APIFootball.com for {date or 'today'}",
            "count": len(fixtures),
            "fixtures": fixtures,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching fixtures from APIFootball.com: {str(e)}"
        )


@router.get("/apifootball/live")
def get_apifootball_live_fixtures():
    """Get live fixtures from APIFootball.com."""
    try:
        service = APIFootballService()
        fixtures = service.get_live_fixtures()

        return {
            "status": "success",
            "message": "Retrieved live fixtures from APIFootball.com",
            "count": len(fixtures),
            "fixtures": fixtures,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching live fixtures from APIFootball.com: {str(e)}"
        )


@router.get("/apifootball/leagues")
def get_apifootball_leagues():
    """Get available leagues from APIFootball.com."""
    try:
        service = APIFootballService()
        leagues = service.get_leagues()

        return {
            "status": "success",
            "message": "Retrieved leagues from APIFootball.com",
            "count": len(leagues),
            "leagues": leagues,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching leagues from APIFootball.com: {str(e)}"
        )


@router.post("/apifootball/sync")
def sync_fixtures_from_apifootball(
    date: Optional[str] = Query(None, description="Date in YYYY-MM-DD format (defaults to today)"),
    db: Session = Depends(get_db)
):
    """Sync fixtures from APIFootball.com to database."""
    try:
        fixture_service = FixtureService(db)
        fixtures = fixture_service.sync_fixtures_from_apifootball(date)

        return {
            "status": "success",
            "message": f"Synced fixtures from APIFootball.com for {date or 'today'}",
            "count": len(fixtures),
            "fixtures": [fixture.to_dict() for fixture in fixtures],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error syncing fixtures from APIFootball.com: {str(e)}"
        )
