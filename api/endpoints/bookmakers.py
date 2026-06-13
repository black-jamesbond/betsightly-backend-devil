"""
Bookmakers API Endpoints

This module defines the API endpoints for bookmakers.
"""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from database import get_db
from bookmaker import Bookmaker
from schemas.bookmaker import BookmakerCreate, BookmakerUpdate, BookmakerResponse, BookmakerListResponse
from utils.common import setup_logging

# Set up logging
logger = setup_logging(__name__)

# Create router
router = APIRouter()

@router.get("/", response_model=BookmakerListResponse)
def get_bookmakers(
    db: Session = Depends(get_db),
    skip: int = Query(0, description="Number of bookmakers to skip"),
    limit: int = Query(100, description="Maximum number of bookmakers to return")
):
    """
    Get all bookmakers.

    Args:
        skip: Number of bookmakers to skip
        limit: Maximum number of bookmakers to return
    """
    try:
        # Get bookmakers
        bookmakers = (
            db.query(Bookmaker)
            .order_by(Bookmaker.name)
            .offset(skip)
            .limit(limit)
            .all()
        )

        # Get total count
        total = db.query(Bookmaker).count()

        # Convert to dictionaries
        bookmakers_dict = [bookmaker.to_dict() for bookmaker in bookmakers]

        return {
            "status": "success",
            "bookmakers": bookmakers_dict,
            "total": total,
            "skip": skip,
            "limit": limit
        }
    except Exception as e:
        logger.error(f"Error getting bookmakers: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting bookmakers: {str(e)}")

@router.get("/{bookmaker_id}", response_model=BookmakerResponse)
def get_bookmaker(
    bookmaker_id: int,
    db: Session = Depends(get_db)
):
    """
    Get bookmaker by ID.

    Args:
        bookmaker_id: Bookmaker ID
    """
    try:
        # Get bookmaker
        bookmaker = db.query(Bookmaker).filter(Bookmaker.id == bookmaker_id).first()

        if not bookmaker:
            raise HTTPException(status_code=404, detail=f"Bookmaker with ID {bookmaker_id} not found")

        return {
            "status": "success",
            "bookmaker": bookmaker.to_dict()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting bookmaker {bookmaker_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting bookmaker: {str(e)}")

@router.post("/", response_model=BookmakerResponse)
def create_bookmaker(
    bookmaker: BookmakerCreate,
    db: Session = Depends(get_db)
):
    """
    Create a new bookmaker.

    Args:
        bookmaker: Bookmaker data
    """
    try:
        # Check if bookmaker already exists
        existing_bookmaker = db.query(Bookmaker).filter(Bookmaker.name == bookmaker.name).first()

        if existing_bookmaker:
            return {
                "status": "success",
                "bookmaker": existing_bookmaker.to_dict()
            }

        # Create new bookmaker
        new_bookmaker = Bookmaker(
            name=bookmaker.name,
            logo_url=bookmaker.logo_url,
            website=bookmaker.website,
            country=bookmaker.country
        )

        # Add to database
        db.add(new_bookmaker)
        db.commit()
        db.refresh(new_bookmaker)

        return {
            "status": "success",
            "bookmaker": new_bookmaker.to_dict()
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating bookmaker: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating bookmaker: {str(e)}")

@router.put("/{bookmaker_id}", response_model=BookmakerResponse)
def update_bookmaker(
    bookmaker_id: int,
    bookmaker: BookmakerUpdate,
    db: Session = Depends(get_db)
):
    """
    Update an existing bookmaker.

    Args:
        bookmaker_id: Bookmaker ID
        bookmaker: Updated bookmaker data
    """
    try:
        # Get bookmaker
        db_bookmaker = db.query(Bookmaker).filter(Bookmaker.id == bookmaker_id).first()

        if not db_bookmaker:
            raise HTTPException(status_code=404, detail=f"Bookmaker with ID {bookmaker_id} not found")

        # Update fields
        update_data = bookmaker.dict(exclude_unset=True)

        for key, value in update_data.items():
            setattr(db_bookmaker, key, value)

        # Update timestamp
        db_bookmaker.updated_at = datetime.now()

        # Commit changes
        db.commit()
        db.refresh(db_bookmaker)

        return {
            "status": "success",
            "bookmaker": db_bookmaker.to_dict()
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating bookmaker {bookmaker_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating bookmaker: {str(e)}")

@router.delete("/{bookmaker_id}")
def delete_bookmaker(
    bookmaker_id: int,
    db: Session = Depends(get_db)
):
    """
    Delete a bookmaker.

    Args:
        bookmaker_id: Bookmaker ID
    """
    try:
        # Get bookmaker
        bookmaker = db.query(Bookmaker).filter(Bookmaker.id == bookmaker_id).first()

        if not bookmaker:
            raise HTTPException(status_code=404, detail=f"Bookmaker with ID {bookmaker_id} not found")

        # Delete bookmaker
        db.delete(bookmaker)
        db.commit()

        return {
            "status": "success",
            "message": f"Bookmaker with ID {bookmaker_id} deleted"
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting bookmaker {bookmaker_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting bookmaker: {str(e)}")

@router.get("/name/{name}")
def get_bookmaker_by_name(
    name: str,
    db: Session = Depends(get_db)
):
    """
    Get bookmaker by name.

    Args:
        name: Bookmaker name
    """
    try:
        # Get bookmaker
        bookmaker = db.query(Bookmaker).filter(Bookmaker.name == name).first()

        if not bookmaker:
            raise HTTPException(status_code=404, detail=f"Bookmaker with name {name} not found")

        return {
            "status": "success",
            "bookmaker": bookmaker.to_dict()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting bookmaker by name {name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting bookmaker: {str(e)}")
