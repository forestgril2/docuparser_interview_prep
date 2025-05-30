# List comprehension
list_squares = [x**2 for x in range(1000000)]
print(f"List memory usage: {list_squares.__sizeof__()} bytes")

# Generator expression
generator_squares = (x**2 for x in range(1000000))
print(f"Generator memory usage: {generator_squares.__sizeof__()} bytes")

# List can be accessed multiple times
print(f"First element of list: {list_squares[0]}")
print(f"First element of list again: {list_squares[0]}")

# Generator can only be consumed once
print(f"First element of generator: {next(generator_squares)}")
print(f"Second element of generator: {next(generator_squares)}")

# Try to access generator elements again
try:
    print(f"First element of generator again: {next(generator_squares)}")
except StopIteration:
    print("Generator has been exhausted!") 