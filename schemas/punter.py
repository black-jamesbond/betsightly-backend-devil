"""
Punter Schemas

This module defines the Pydantic schemas for punters.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict

class PunterBase(BaseModel):
    """Base schema for punter data."""
    name: str = Field(..., description="Punter name")
    telegram_username: Optional[str] = Field(None, description="Telegram username")
    telegram_channel: Optional[str] = Field(None, description="Telegram channel")
    twitter_username: Optional[str] = Field(None, description="Twitter username")
    bio: Optional[str] = Field(None, description="Punter biography")
    avatar_url: Optional[str] = Field(None, description="URL to punter avatar")
    specialties: Optional[List[str]] = Field(None, description="List of prediction specialties")

class PunterCreate(PunterBase):
    """Schema for creating a punter."""
    pass

class PunterUpdate(BaseModel):
    """Schema for updating a punter."""
    name: Optional[str] = Field(None, description="Punter name")
    telegram_username: Optional[str] = Field(None, description="Telegram username")
    telegram_channel: Optional[str] = Field(None, description="Telegram channel")
    twitter_username: Optional[str] = Field(None, description="Twitter username")
    bio: Optional[str] = Field(None, description="Punter biography")
    avatar_url: Optional[str] = Field(None, description="URL to punter avatar")
    specialties: Optional[List[str]] = Field(None, description="List of prediction specialties")

class PunterInDB(PunterBase):
    """Schema for punter data in the database."""
    id: str = Field(..., description="Punter ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")

    model_config = ConfigDict(from_attributes=True)

class PunterResponse(BaseModel):
    """Schema for punter response."""
    status: str = Field(..., description="Response status")
    punter: Dict[str, Any] = Field(..., description="Punter data")

class PunterListResponse(BaseModel):
    """Schema for punter list response."""
    status: str = Field(..., description="Response status")
    punters: List[Dict[str, Any]] = Field(..., description="List of punters")
    total: int = Field(..., description="Total number of punters")
    skip: int = Field(..., description="Number of punters skipped")
    limit: int = Field(..., description="Maximum number of punters returned")
