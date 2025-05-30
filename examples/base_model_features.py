from pydantic import BaseModel, ValidationError
from typing import Dict, Any

class SimpleModel(BaseModel):
    name: str
    value: int

# 1. Automatic validation
try:
    # This will fail because 'value' is not an integer
    model = SimpleModel(name="test", value="not an int")
except ValidationError as e:
    print("1. Validation Error:", e)

# 2. Data conversion
model = SimpleModel(name="test", value="123")  # String "123" is converted to int
print("\n2. Data Conversion:", model)

# 3. Dictionary conversion
data_dict = model.model_dump()
print("\n3. To Dict:", data_dict)

# 4. JSON conversion
json_data = model.model_dump_json()
print("\n4. To JSON:", json_data)

# 5. Field access
print("\n5. Field Access:")
print(f"   name: {model.name}")
print(f"   value: {model.value}")

# 6. Field types
print("\n6. Field Types:")
print(f"   name type: {type(model.name)}")
print(f"   value type: {type(model.value)}")

# 7. Model configuration
class ConfigModel(BaseModel):
    name: str
    value: int

    class Config:
        # Allow extra fields
        extra = "allow"
        # Make fields immutable
        frozen = True
        # Allow population by field name
        allow_population_by_field_name = True

# 8. Extra fields handling
config_model = ConfigModel(name="test", value=123, extra_field="allowed")
print("\n8. Extra Fields:", config_model.model_dump())

# 9. Field validation
class ValidationModel(BaseModel):
    name: str
    value: int

    def model_post_init(self, __context: Any) -> None:
        """Called after model initialization"""
        if self.value < 0:
            raise ValueError("Value must be positive")

try:
    ValidationModel(name="test", value=-1)
except ValueError as e:
    print("\n9. Custom Validation:", e)

# 10. Model methods
print("\n10. Model Methods:")
print(f"   model_dump(): {model.model_dump()}")
print(f"   model_dump_json(): {model.model_dump_json()}")
print(f"   model_fields: {model.model_fields}")
print(f"   model_config: {model.model_config}") 