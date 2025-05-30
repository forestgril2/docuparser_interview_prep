class Example:
    # Regular instance method - needs @classmethod decorator to use cls
    @classmethod
    def regular_class_method(cls):
        return f"This is a class method of {cls.__name__}"
    
    # __new__ is automatically a class method - no decorator needed
    def __new__(cls):
        print(f"Creating new instance of {cls.__name__}")
        return super().__new__(cls)
    
    # Regular instance method - uses self
    def instance_method(self):
        return f"This is an instance method of {self.__class__.__name__}"

# Using the methods
print(Example.regular_class_method())  # Needs @classmethod decorator
print(Example())  # __new__ is automatically called, no decorator needed
print(Example().instance_method())  # Regular instance method 