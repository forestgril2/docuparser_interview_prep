class MathOperations:
    PI = 3.14159  # Class variable
    
    def __init__(self, value):
        self.value = value
    
    # Static method - no access to class or instance
    @staticmethod
    def add(x, y):
        return x + y
    
    # Class method - has access to class but not instance
    @classmethod
    def get_pi(cls):
        return cls.PI
    
    # Instance method - has access to both class and instance
    def multiply_by_pi(self):
        return self.value * self.PI

# Using static method - no instance needed
print(f"Static method: {MathOperations.add(5, 3)}")  # 8

# Using class method - no instance needed
print(f"Class method: {MathOperations.get_pi()}")  # 3.14159

# Using instance method - needs instance
math_ops = MathOperations(2)
print(f"Instance method: {math_ops.multiply_by_pi()}")  # 6.28318

# Can also use static method on instance (but not recommended)
print(f"Static method on instance: {math_ops.add(5, 3)}")  # 8 