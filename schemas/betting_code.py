"""
Betting Code Schemas

This module defines the Pydantic schemas for betting codes.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict

class BettingCodeBase(BaseModel):
    """Base schema for betting code data."""
    code: str = Field(..., description="Betting/booking code")
    punter_id: int = Field(..., description="ID of the punter who provided the code")
    bookmaker_id: Optional[int] = Field(None, description="ID of the bookmaker")
    odds: Optional[float] = Field(None, description="Odds value")
    event_date: Optional[datetime] = Field(None, description="Date of the event")
    expiry_date: Optional[datetime] = Field(None, description="Expiry date of the code")
    status: Optional[str] = Field("pending", description="Status of the code (pending, won, lost)")
    confidence: Optional[int] = Field(None, description="Confidence level (1-10)")
    featured: Optional[bool] = Field(False, description="Whether the code is featured")
    notes: Optional[str] = Field(None, description="Additional notes")

class BettingCodeCreate(BettingCodeBase):
    """Schema for creating a betting code."""
    pass

class BettingCodeUpdate(BaseModel):
    """Schema for updating a betting code."""
    code: Optional[str] = Field(None, description="Betting/booking code")
    punter_id: Optional[int] = Field(None, description="ID of the punter who provided the code")
    bookmaker_id: Optional[int] = Field(None, description="ID of the bookmaker")
    odds: Optional[float] = Field(None, description="Odds value")
    event_date: Optional[datetime] = Field(None, description="Date of the event")
    expiry_date: Optional[datetime] = Field(None, description="Expiry date of the code")
    status: Optional[str] = Field(None, description="Status of the code (pending, won, lost)")
    confidence: Optional[int] = Field(None, description="Confidence level (1-10)")
    featured: Optional[bool] = Field(None, description="Whether the code is featured")
    notes: Optional[str] = Field(None, description="Additional notes")

class BettingCodeInDB(BettingCodeBase):
    """Schema for betting code data in the database."""
    id: int = Field(..., description="Betting code ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")

    model_config = ConfigDict(from_attributes=True)

class BettingCodeResponse(BaseModel):
    """Schema for betting code response."""
    status: str = Field(..., description="Response status")
    betting_code: Dict[str, Any] = Field(..., description="Betting code data")

class BettingCodeListResponse(BaseModel):
    """Schema for betting code list response."""
    status: str = Field(..., description="Response status")
    betting_codes: List[Dict[str, Any]] = Field(..., description="List of betting codes")
    total: int = Field(..., description="Total number of betting codes")
    skip: int = Field(..., description="Number of betting codes skipped")
    limit: int = Field(..., description="Maximum number of betting codes returned")
