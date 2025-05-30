from pydantic import BaseModel, Field, EmailStr, validator
from typing import List, Optional
from datetime import datetime

class User(BaseModel):
    # Basic fields with validation
    id: int
    name: str = Field(..., min_length=2, max_length=50)
    email: EmailStr
    age: int = Field(..., gt=0, lt=120)
    
    # Optional fields
    phone: Optional[str] = None
    
    # Nested models
    addresses: List['Address'] = []
    
    # Custom validation
    @validator('name')
    def name_must_contain_space(cls, v):
        if ' ' not in v:
            raise ValueError('name must contain a space')
        return v.title()
    
    # Config for additional behavior
    class Config:
        json_schema_extra = {
            "example": {
                "id": 1,
                "name": "John Doe",
                "email": "john@example.com",
                "age": 30,
                "phone": "+1234567890",
                "addresses": []
            }
        }

class Address(BaseModel):
    street: str
    city: str
    country: str
    postal_code: str

# Example usage
try:
    # Valid data
    user_data = {
        "id": 1,
        "name": "John Doe",
        "email": "john@example.com",
        "age": 30,
        "addresses": [
            {
                "street": "123 Main St",
                "city": "New York",
                "country": "USA",
                "postal_code": "10001"
            }
        ]
    }
    user = User(**user_data)
    print("Valid user created:", user.model_dump())

    # Invalid data - will raise validation error
    invalid_data = {
        "id": 2,
        "name": "Jane",  # No space in name
        "email": "invalid-email",  # Invalid email
        "age": 150  # Age out of range
    }
    invalid_user = User(**invalid_data)
except Exception as e:
    print("Validation error:", str(e))

# Convert to/from JSON
json_data = user.model_dump_json()
print("\nJSON representation:", json_data)

# Create from JSON
new_user = User.model_validate_json(json_data)
print("\nRecreated from JSON:", new_user.model_dump()) 