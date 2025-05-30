from typing import ClassVar

class Document:
    # Class-level type hint (shared by all instances)
    default_title: ClassVar[str] = "Untitled"
    
    # Instance-level type hint (unique to each instance)
    def __init__(self, title: str):
        self.title: str = title  # Instance attribute with type hint
    
    # Method parameter type hint
    def set_title(self, new_title: str) -> None:
        self.title = new_title

# Using the class
doc1 = Document("First Document")
doc2 = Document("Second Document")

print(f"Class variable: {Document.default_title}")  # Shared by all instances
print(f"Instance 1 title: {doc1.title}")  # Unique to doc1
print(f"Instance 2 title: {doc2.title}")  # Unique to doc2

# Type hints help with IDE support and type checking
doc1.set_title("New Title")  # IDE will show type hints for parameter 