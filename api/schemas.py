from typing import List, Optional
from pydantic import BaseModel, Field


class TransactionInput(BaseModel):
    """Schema for single incoming transaction inference requests."""

    TransactionID: int = Field(..., json_schema_extra={"example": 2987000})
    TransactionDT: int = Field(..., json_schema_extra={"example": 86400})
    TransactionAmt: float = Field(..., json_schema_extra={"example": 68.5})
    ProductCD: str = Field(..., json_schema_extra={"example": "W"})
    card1: int = Field(..., json_schema_extra={"example": 13926})
    card2: Optional[float] = Field(None, json_schema_extra={"example": 360.0})
    card3: Optional[float] = Field(None, json_schema_extra={"example": 150.0})
    card4: Optional[str] = Field(None, json_schema_extra={"example": "discover"})
    card5: Optional[float] = Field(None, json_schema_extra={"example": 142.0})
    card6: Optional[str] = Field(None, json_schema_extra={"example": "credit"})
    addr1: Optional[float] = Field(None, json_schema_extra={"example": 315.0})
    addr2: Optional[float] = Field(None, json_schema_extra={"example": 87.0})
    P_emaildomain: Optional[str] = Field(None, json_schema_extra={"example": "gmail.com"})
    R_emaildomain: Optional[str] = Field(None, json_schema_extra={"example": "gmail.com"})


class PredictionOutput(BaseModel):
    """Schema for single transaction prediction response."""

    TransactionID: int
    fraud_probability: float
    is_fraud: bool


class BatchTransactionInput(BaseModel):
    """Schema for batch inference requests."""

    transactions: List[TransactionInput]
