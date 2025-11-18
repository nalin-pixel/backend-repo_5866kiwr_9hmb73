"""
Database Schemas for Trade Journal SaaS

Each Pydantic model maps to a MongoDB collection (lowercased class name).
"""
from typing import Optional, List, Literal
from pydantic import BaseModel, Field, EmailStr
from datetime import datetime

class User(BaseModel):
    email: EmailStr = Field(..., description="Unique email")
    password_hash: str = Field(..., description="BCrypt password hash")
    name: Optional[str] = Field(None, description="Display name")
    subscription_status: Literal['none','active','past_due','canceled'] = 'none'
    stripe_customer_id: Optional[str] = None
    stripe_subscription_id: Optional[str] = None

class Trade(BaseModel):
    user_id: str = Field(..., description="Owner user id (stringified ObjectId)")
    ticker: str = Field(..., description="Symbol, e.g., AAPL")
    entry_price: float = Field(..., ge=0)
    exit_price: Optional[float] = Field(None, ge=0)
    position_size: float = Field(..., ge=0)
    direction: Literal['long','short'] = 'long'
    date: datetime = Field(default_factory=datetime.utcnow)
    setup: Optional[str] = None
    reasoning: Optional[str] = None
    emotions: Optional[str] = None
    screenshots: Optional[List[str]] = Field(default=None, description="URLs to uploaded screenshots")
    outcome: Optional[Literal['win','loss','breakeven']] = None

class Feedback(BaseModel):
    user_id: str
    trade_id: str
    positives: List[str] = []
    negatives: List[str] = []
    suggestions: List[str] = []
    summary: Optional[str] = None

class Subscription(BaseModel):
    user_id: str
    status: Literal['active','past_due','canceled','incomplete','trialing'] = 'incomplete'
    stripe_customer_id: Optional[str] = None
    stripe_subscription_id: Optional[str] = None
