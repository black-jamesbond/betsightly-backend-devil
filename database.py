"""
Database configuration for the application.
"""

import logging
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Import settings after loading environment
from utils.config import settings

# Get database URL from settings
DATABASE_URL = settings.database.URL

# Determine if we're using SQLite or PostgreSQL
if DATABASE_URL.startswith("sqlite"):
    DB_FILE = DATABASE_URL.replace("sqlite:///", "")
    logger.info(f"Using SQLite database: {DB_FILE}")
    # Create engine with SQLite-specific options
    engine = create_engine(
        DATABASE_URL, connect_args={"check_same_thread": False}
    )
else:
    # PostgreSQL or other database
    logger.info(f"Using PostgreSQL database: {DATABASE_URL}")
    # Create engine without SQLite-specific options
    engine = create_engine(DATABASE_URL)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()

def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initialize database with the correct schema."""
    try:
        # Import models in correct order to avoid circular dependencies
        from punter import Punter  # noqa: F401
        from bookmaker import Bookmaker  # noqa: F401
        from betting_code import BettingCode  # noqa: F401
        from fixture import Fixture  # noqa: F401
        from prediction import Prediction  # noqa: F401

        # Create SQLAlchemy tables (works for both SQLite and PostgreSQL)
        Base.metadata.create_all(bind=engine)
        logger.info("SQLAlchemy tables created.")

        # Apply database optimizations
        try:
            from utils.database_optimization import create_database_indexes, optimize_query_performance
            create_database_indexes()
            optimize_query_performance()
            logger.info("Database optimizations applied.")
        except ImportError:
            logger.warning("Database optimization utilities not available")
        except Exception as e:
            logger.warning(f"Failed to apply database optimizations: {str(e)}")

        logger.info("Database initialized successfully")
        return True

    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        return False
