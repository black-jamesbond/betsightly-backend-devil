"""
Fixture service for managing football match fixtures.
Enhanced with APIFootball.com integration for testing.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session

from fixture import Fixture
from services.apifootball_service import APIFootballService

logger = logging.getLogger(__name__)
from prediction import Prediction

class FixtureService:
    """Service for managing football match fixtures."""

    def __init__(self, db: Session):
        """Initialize with database session."""
        self.db = db
        self.apifootball_service = APIFootballService()

    def get_fixture_by_id(self, fixture_id: int) -> Optional[Fixture]:
        """Get a fixture by ID."""
        return self.db.query(Fixture).filter(Fixture.fixture_id == fixture_id).first()

    def get_fixtures_by_date(self, date: datetime) -> List[Fixture]:
        """Get fixtures by date."""
        start_date = datetime(date.year, date.month, date.day, 0, 0, 0)
        end_date = start_date + timedelta(days=1)
        return self.db.query(Fixture).filter(
            Fixture.date >= start_date,
            Fixture.date < end_date
        ).all()

    def get_fixtures_by_league(self, league_id: int, date: Optional[datetime] = None) -> List[Fixture]:
        """Get fixtures by league."""
        query = self.db.query(Fixture).filter(Fixture.league_id == league_id)

        if date:
            start_date = datetime(date.year, date.month, date.day, 0, 0, 0)
            end_date = start_date + timedelta(days=1)
            query = query.filter(
                Fixture.date >= start_date,
                Fixture.date < end_date
            )

        return query.all()

    def create_fixture(self, fixture_data: Dict[str, Any]) -> Fixture:
        """Create a new fixture."""
        fixture = Fixture.from_api(fixture_data)
        self.db.add(fixture)
        self.db.commit()
        self.db.refresh(fixture)
        return fixture

    def update_fixture(self, fixture_id: int, fixture_data: Dict[str, Any]) -> Optional[Fixture]:
        """Update an existing fixture."""
        fixture = self.get_fixture_by_id(fixture_id)
        if not fixture:
            return None

        for key, value in fixture_data.items():
            if hasattr(fixture, key):
                setattr(fixture, key, value)

        self.db.commit()
        self.db.refresh(fixture)
        return fixture

    def delete_fixture(self, fixture_id: int) -> bool:
        """Delete a fixture."""
        fixture = self.get_fixture_by_id(fixture_id)
        if not fixture:
            return False

        self.db.delete(fixture)
        self.db.commit()
        return True

    def create_or_update_fixtures(self, fixtures_data: List[Dict[str, Any]]) -> List[Fixture]:
        """Create or update fixtures from API data."""
        fixtures = []

        for fixture_data in fixtures_data:
            # Convert date string to datetime if needed
            if "date" in fixture_data and isinstance(fixture_data["date"], str):
                try:
                    fixture_data["date"] = datetime.fromisoformat(fixture_data["date"].replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    # If date conversion fails, use current time
                    fixture_data["date"] = datetime.utcnow()

            existing_fixture = self.get_fixture_by_id(fixture_data["fixture_id"])

            if existing_fixture:
                # Update existing fixture
                for key, value in fixture_data.items():
                    if hasattr(existing_fixture, key):
                        setattr(existing_fixture, key, value)
                fixtures.append(existing_fixture)
            else:
                # Create new fixture
                try:
                    fixture = Fixture.from_api(fixture_data)
                except Exception as e:
                    # If from_api fails, create fixture directly
                    fixture = Fixture(
                        fixture_id=fixture_data["fixture_id"],
                        date=fixture_data["date"],
                        league_id=fixture_data["league_id"],
                        league_name=fixture_data["league_name"],
                        home_team_id=fixture_data["home_team_id"],
                        home_team=fixture_data["home_team"],
                        away_team_id=fixture_data["away_team_id"],
                        away_team=fixture_data["away_team"],
                        home_odds=fixture_data.get("home_odds", 0),
                        draw_odds=fixture_data.get("draw_odds", 0),
                        away_odds=fixture_data.get("away_odds", 0)
                    )
                self.db.add(fixture)
                fixtures.append(fixture)

        self.db.commit()
        for fixture in fixtures:
            self.db.refresh(fixture)

        return fixtures

    def get_fixtures_from_apifootball(self, date: str = None) -> List[Dict[str, Any]]:
        """
        Get fixtures from APIFootball.com for testing.

        Args:
            date: Date in YYYY-MM-DD format (defaults to today)

        Returns:
            List of fixture dictionaries
        """
        try:
            logger.info(f"🌐 Fetching fixtures from APIFootball.com for {date or 'today'}")
            fixtures = self.apifootball_service.get_daily_fixtures(date)
            logger.info(f"✅ Retrieved {len(fixtures)} fixtures from APIFootball.com")
            return fixtures
        except Exception as e:
            logger.error(f"❌ Error fetching fixtures from APIFootball.com: {str(e)}")
            return []

    def get_live_fixtures_from_apifootball(self) -> List[Dict[str, Any]]:
        """
        Get live fixtures from APIFootball.com.

        Returns:
            List of live fixture dictionaries
        """
        try:
            logger.info("🔴 Fetching live fixtures from APIFootball.com")
            fixtures = self.apifootball_service.get_live_fixtures()
            logger.info(f"✅ Retrieved {len(fixtures)} live fixtures from APIFootball.com")
            return fixtures
        except Exception as e:
            logger.error(f"❌ Error fetching live fixtures from APIFootball.com: {str(e)}")
            return []

    def sync_fixtures_from_apifootball(self, date: str = None) -> List[Fixture]:
        """
        Sync fixtures from APIFootball.com to database.

        Args:
            date: Date in YYYY-MM-DD format (defaults to today)

        Returns:
            List of created/updated Fixture objects
        """
        try:
            logger.info(f"🔄 Syncing fixtures from APIFootball.com for {date or 'today'}")

            # Get fixtures from APIFootball.com
            api_fixtures = self.get_fixtures_from_apifootball(date)

            if not api_fixtures:
                logger.warning("⚠️  No fixtures retrieved from APIFootball.com")
                return []

            # Convert and store in database
            db_fixtures = []

            for api_fixture in api_fixtures:
                try:
                    # Check if fixture already exists
                    existing_fixture = self.db.query(Fixture).filter(
                        Fixture.fixture_id == api_fixture["fixture_id"]
                    ).first()

                    if existing_fixture:
                        # Update existing fixture
                        existing_fixture.home_team = api_fixture["home_team"]
                        existing_fixture.away_team = api_fixture["away_team"]
                        existing_fixture.date = datetime.fromisoformat(api_fixture["date"].replace('Z', '+00:00'))
                        existing_fixture.league = api_fixture["league_name"]
                        existing_fixture.home_odds = api_fixture.get("home_odds", 0)
                        existing_fixture.draw_odds = api_fixture.get("draw_odds", 0)
                        existing_fixture.away_odds = api_fixture.get("away_odds", 0)
                        db_fixtures.append(existing_fixture)
                        logger.debug(f"📝 Updated fixture: {existing_fixture.home_team} vs {existing_fixture.away_team}")
                    else:
                        # Create new fixture
                        new_fixture = Fixture(
                            fixture_id=api_fixture["fixture_id"],
                            home_team=api_fixture["home_team"],
                            away_team=api_fixture["away_team"],
                            date=datetime.fromisoformat(api_fixture["date"].replace('Z', '+00:00')),
                            league=api_fixture["league_name"],
                            home_odds=api_fixture.get("home_odds", 0),
                            draw_odds=api_fixture.get("draw_odds", 0),
                            away_odds=api_fixture.get("away_odds", 0)
                        )
                        self.db.add(new_fixture)
                        db_fixtures.append(new_fixture)
                        logger.debug(f"➕ Created fixture: {new_fixture.home_team} vs {new_fixture.away_team}")

                except Exception as e:
                    logger.warning(f"⚠️  Error processing fixture {api_fixture.get('fixture_id', 'unknown')}: {str(e)}")
                    continue

            # Commit changes
            self.db.commit()

            # Refresh objects
            for fixture in db_fixtures:
                self.db.refresh(fixture)

            logger.info(f"✅ Successfully synced {len(db_fixtures)} fixtures from APIFootball.com")
            return db_fixtures

        except Exception as e:
            logger.error(f"❌ Error syncing fixtures from APIFootball.com: {str(e)}")
            self.db.rollback()
            return []

    def test_apifootball_connection(self) -> bool:
        """
        Test connection to APIFootball.com.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            return self.apifootball_service.test_connection()
        except Exception as e:
            logger.error(f"❌ Error testing APIFootball.com connection: {str(e)}")
            return False
