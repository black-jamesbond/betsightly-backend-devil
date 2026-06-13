"""
Bookmaker Schemas

This module defines the Pydantic schemas for bookmakers.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict

class BookmakerBase(BaseModel):
    """Base schema for bookmaker data."""
    name: str = Field(..., description="Bookmaker name")
    logo_url: Optional[str] = Field(None, description="URL to bookmaker logo")
    website: Optional[str] = Field(None, description="Bookmaker website")
    country: Optional[str] = Field(None, description="Bookmaker's country")

class BookmakerCreate(BookmakerBase):
    """Schema for creating a bookmaker."""
    pass

class BookmakerUpdate(BaseModel):
    """Schema for updating a bookmaker."""
    name: Optional[str] = Field(None, description="Bookmaker name")
    logo_url: Optional[str] = Field(None, description="URL to bookmaker logo")
    website: Optional[str] = Field(None, description="Bookmaker website")
    country: Optional[str] = Field(None, description="Bookmaker's country")

class BookmakerInDB(BookmakerBase):
    """Schema for bookmaker data in the database."""
    id: int = Field(..., description="Bookmaker ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")

    model_config = ConfigDict(from_attributes=True)

class BookmakerResponse(BaseModel):
    """Schema for bookmaker response."""
    status: str = Field(..., description="Response status")
    bookmaker: Dict[str, Any] = Field(..., description="Bookmaker data")

class BookmakerListResponse(BaseModel):
    """Schema for bookmaker list response."""
    status: str = Field(..., description="Response status")
    bookmakers: List[Dict[str, Any]] = Field(..., description="List of bookmakers")
    total: int = Field(..., description="Total number of bookmakers")
    skip: int = Field(..., description="Number of bookmakers skipped")
    limit: int = Field(..., description="Maximum number of bookmakers returned")
