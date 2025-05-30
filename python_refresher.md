# Python Interview Refresher

This is a focused 2h45min Python refresher targeting the most commonly asked backend Python interview questions.

## Time Management Plan
- 45 min: Core Python Fundamentals
- 30 min: Context Managers & Resource Management
- 45 min: Async Programming & Concurrency  
- 30 min: OOP & Design Patterns
- 30 min: Error Handling & Exceptions
- 45 min: Backend Specifics (FastAPI/Flask/Database)
- 30 min: Practice Problems
- 45 min: Relax before interview

---

## Block 1: Core Python Fundamentals (45 minutes)

### Data Structures & Algorithms (Asked 90% of the time)
```python
# 1. List Comprehensions vs Generators
numbers = [1, 2, 3, 4, 5]
squares_list = [x**2 for x in numbers]  # List - memory intensive
squares_gen = (x**2 for x in numbers)   # Generator - memory efficient
# Memory Efficiency Explanation:
# List comprehension creates the entire list in memory at once
# Generator expression creates values on-demand, one at a time
# Example:
numbers = [1, 2, 3, 4, 5]
list_memory = [x**2 for x in numbers]  # Creates [1, 4, 9, 16, 25] all at once
gen_memory = (x**2 for x in numbers)   # Creates values only when needed

# Memory usage comparison:
import sys
print(f"List memory: {sys.getsizeof(list_memory)} bytes")
print(f"Generator memory: {sys.getsizeof(gen_memory)} bytes")
# The prints will show:
# List memory: 96 bytes
# Generator memory: 112 bytes

# Explanation:
# - The list memory size (96 bytes) includes the actual data (integers) plus list overhead
# - The generator memory size (112 bytes) is just the generator object overhead
# - For small datasets, generators might actually use more memory due to their object overhead
# - The real memory benefit comes with large datasets where generators don't store all values


# For large datasets, this difference becomes significant:
large_numbers = range(1000000)
large_list = [x**2 for x in large_numbers]  # Could cause memory issues
large_gen = (x**2 for x in large_numbers)   # Memory efficient

# Why use lists instead of generators?
# 1. Random access - lists allow indexing (my_list[5])
# 2. Multiple iterations - generators can only be iterated once
# 3. Length determination - len(my_list) works, len(my_gen) doesn't
# 4. List operations - sorting, reversing, etc. require full data
# 5. Debugging - easier to inspect list contents
# 6. Small datasets - memory overhead is negligible



# When to use each:
# List: Need random access, multiple iterations
# Generator: Large datasets, single iteration, memory efficiency


# 2. Dictionary Operations (very common)
# Merge dictionaries (Python 3.9+)
dict1 = {'a': 1, 'b': 2}
dict2 = {'c': 3, 'd': 4}
merged = dict1 | dict2

# Get with default
result = dict1.get('x', 'default_value')

# Dictionary comprehension
squares_dict = {x: x**2 for x in range(5)}
# Dictionary generators (similar to list comprehensions)
# Basic dictionary generator
squares = {x: x**2 for x in range(5)}  # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

# With conditional
even_squares = {x: x**2 for x in range(10) if x % 2 == 0}  # {0: 0, 2: 4, 4: 16, 6: 36, 8: 64}

# Nested dictionary generator
matrix = {(i, j): i*j for i in range(3) for j in range(3)}
# {(0,0): 0, (0,1): 0, (0,2): 0, (1,0): 0, (1,1): 1, (1,2): 2, (2,0): 0, (2,1): 2, (2,2): 4}

# Dictionary generator with string keys
word_lengths = {word: len(word) for word in ['apple', 'banana', 'cherry']}
# {'apple': 5, 'banana': 6, 'cherry': 6}

# Dictionary generator with value transformation
original = {'a': 1, 'b': 2, 'c': 3}
doubled = {k: v*2 for k, v in original.items()}
# {'a': 2, 'b': 4, 'c': 6}

# 3. Set Operations
set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}
intersection = set1 & set2  # {3, 4}
union = set1 | set2         # {1, 2, 3, 4, 5, 6}
difference = set1 - set2    # {1, 2}
```

### String Manipulation (Asked 80% of the time)
```python

# 1. String formatting
name, age = "John", 25
f_string = f"Name: {name}, Age: {age}"
format_method = "Name: {}, Age: {}".format(name, age)
percent_format = "Name: %s, Age: %d" % (name, age)
# Format method vs f-strings vs % formatting

# 1. Format method (.format())
# - More flexible with named parameters
# - Can reuse parameters
# - Better for complex formatting
# - Slower than f-strings
# - More verbose syntax

# Example with named parameters
format_named = "Name: {name}, Age: {age}, Name again: {name}".format(name="John", age=25)

# Example with positional parameters
format_positional = "Name: {}, Age: {}, Name again: {}".format("John", 25, "John")

# Example with index
format_index = "Name: {0}, Age: {1}, Name again: {0}".format("John", 25)

# 2. % formatting (old style)
# - Similar to C-style printf
# - Less flexible than .format()
# - Being phased out in favor of f-strings
# - Can be confusing with multiple parameters
# - Faster than .format() but slower than f-strings

# Example with named parameters
percent_named = "Name: %(name)s, Age: %(age)d" % {"name": "John", "age": 25}

# Example with positional parameters
percent_positional = "Name: %s, Age: %d" % ("John", 25)

# 3. Key differences from f-strings:
# - f-strings are evaluated at runtime
# - f-strings can contain expressions
# - f-strings are more readable
# - f-strings are faster
# - f-strings require Python 3.6+

# Example showing f-string expression evaluation
x = 10
f_string_expr = f"Value: {x}, Double: {x*2}, Square: {x**2}"

# Cannot do this with .format() or % formatting
# f-strings allow direct expression evaluation

# 2. String operations
text = "  Hello World  "
cleaned = text.strip().lower().replace(" ", "_")  # "hello_world"

# 3. Join vs concatenation
words = ['hello', 'world', 'python']
efficient = '_'.join(words)  # Preferred way
inefficient = words[0] + '_' + words[1] + '_' + words[2]  # Avoid

# 4. Regular expressions
import re
pattern = r'\d+'
text = "There are 123 apples and 456 oranges"
numbers = re.findall(pattern, text)  # ['123', '456']
```
# 5. Vim-style replacement with re
text = "The quick brown fox jumps over the lazy dog"
# Replace 'fox' with 'cat' globally
replaced = re.sub(r'fox', 'cat', text)  # "The quick brown cat jumps over the lazy dog"
# Replace with count limit
replaced_once = re.sub(r'fox', 'cat', text, count=1)  # "The quick brown cat jumps over the lazy dog"
# Case-insensitive replacement
replaced_case = re.sub(r'FOX', 'cat', text, flags=re.IGNORECASE)  # "The quick brown cat jumps over the lazy dog"



## Block 2: Context Managers & Resource Management (30 minutes)

### Context Managers and with/as Blocks (Asked 75% of the time)
```python
# Context managers are objects that implement the context manager protocol
# (__enter__ and __exit__ methods) to manage resources properly

# Basic context manager example
class FileManager:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.file = None
    
    def __enter__(self):
        self.file = open(self.filename, self.mode)
        return self.file
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()
        # Return False to propagate exceptions
        return False

# Usage with with/as
with FileManager('example.txt', 'w') as f:
    f.write('Hello, World!')
# File is automatically closed after the block

# Common built-in context managers
with open('file.txt', 'r') as f:
    content = f.read()

# Multiple context managers
with open('input.txt', 'r') as source, open('output.txt', 'w') as dest:
    dest.write(source.read())

# Context managers ensure proper resource cleanup
# even if exceptions occur within the with block
```

### Using contextlib (Critical for resource management)
```python
from contextlib import contextmanager, closing
import sqlite3

# 1. @contextmanager decorator
# The @contextmanager decorator transforms a generator function into a context manager
# It handles the __enter__ and __exit__ methods automatically, making it easier
# to create context managers without writing a full class
# Decorator Basics
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("Something is happening before the function is called.")
        result = func(*args, **kwargs)
        print("Something is happening after the function is called.")
        return result
    return wrapper

@my_decorator
def say_hello(name):
    print(f"Hello, {name}!")
# Using the @my_decorator
say_hello("Alice")  # Will print:
# Something is happening before the function is called.
# Hello, Alice!
# Something is happening after the function is called.



# Using the @Timer decorator
slow_function()  # Will print:
# Function slow_function called 1 times
# Function executed

# Decorator with Arguments
def repeat(times):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(times=3)
def greet(name):
    print(f"Greetings, {name}!")

# Using the @repeat decorator
greet("Bob")  # Will print:
# Greetings, Bob!
# Greetings, Bob!
# Greetings, Bob!

# Preserving Function Metadata
from functools import wraps

def preserve_metadata(func):
    @wraps(func)  # Preserves the original function's metadata
    def wrapper(*args, **kwargs):
        print("Decorator is running")
        return func(*args, **kwargs)
    return wrapper


@preserve_metadata
def example_function():
    """This is a docstring"""
    pass
# Using the @preserve_metadata decorator
print(example_function.__name__)  # Will print: example_function
print(example_function.__doc__)   # Will print: This is a docstring

# Class-based Decorator
class Timer:
    def __init__(self, func):
        self.func = func
        self.times_called = 0
    
    def __call__(self, *args, **kwargs):
        self.times_called += 1
        print(f"Function {self.func.__name__} called {self.times_called} times")
        return self.func(*args, **kwargs)

@Timer
def slow_function():
    import time
    time.sleep(1)
    print("Function executed")

@contextmanager
def database_connection(db_name):
    conn = sqlite3.connect(db_name)
    try:
        yield conn
    finally:
        conn.close()

# Usage
with database_connection('example.db') as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users")
    results = cursor.fetchall()

# 2. contextlib.closing for objects with close() method
import urllib.request

with closing(urllib.request.urlopen('http://example.com')) as response:
    data = response.read()

# 3. ExitStack for dynamic context managers
from contextlib import ExitStack

files = ['file1.txt', 'file2.txt', 'file3.txt']
with ExitStack() as stack:
    opened_files = [
        stack.enter_context(open(fname, 'r'))
        for fname in files
    ]
    # Process all files
    for f in opened_files:
        content = f.read()
```

### Custom Context Managers for Backend Applications
```python
# Database transaction context manager
class DatabaseTransaction:
    def __init__(self, connection):
        self.connection = connection
        self.transaction = None
    
    def __enter__(self):
        self.transaction = self.connection.begin()
        return self.transaction
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.transaction.commit()
        else:
            self.transaction.rollback()
        return False

# Redis lock context manager
class RedisLock:
    def __init__(self, redis_client, key, timeout=10):
        self.redis = redis_client
        self.key = key
        self.timeout = timeout
        self.lock_acquired = False
    
    def __enter__(self):
        self.lock_acquired = self.redis.set(
            self.key, "locked", nx=True, ex=self.timeout
        )
        if not self.lock_acquired:
            raise Exception("Could not acquire lock")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.lock_acquired:
            self.redis.delete(self.key)
        return False

# Usage in document processing
async def process_document_safely(document_id, redis_client, db_connection):
    lock_key = f"document_lock_{document_id}"
    
    with RedisLock(redis_client, lock_key):
        with DatabaseTransaction(db_connection):
            # Process document safely
            pass
```

---

## Block 3: Async Programming & Concurrency (45 minutes)

### async/await (Critical for backend)
```python
import asyncio
import aiohttp
import time

# 1. Basic async function
async def fetch_data(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

# 2. Running multiple async tasks
async def fetch_multiple():
    urls = ['http://example1.com', 'http://example2.com']
    tasks = [fetch_data(url) for url in urls]
    results = await asyncio.gather(*tasks)
    return results

# 3. Async context manager (commonly asked)
class AsyncResourceManager:
    async def __aenter__(self):
        print("Acquiring resource")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        print("Releasing resource")
        return False

# Usage
async def use_resource():
    async with AsyncResourceManager() as manager:
        # Do work with resource
        pass

# 4. Async iterator
class AsyncCounter:
    def __init__(self, max_count):
        self.max_count = max_count
        self.count = 0
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        if self.count >= self.max_count:
            raise StopAsyncIteration
        self.count += 1
        await asyncio.sleep(0.1)  # Simulate async work
        return self.count

# Usage
async def use_async_iterator():
    async for number in AsyncCounter(5):
        print(number)
```

### Threading vs Multiprocessing (Asked 70% of the time)
```python
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# 1. Threading for I/O bound tasks
def io_bound_task(n):
    time.sleep(1)  # Simulating I/O
    return n * 2

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(io_bound_task, range(5)))

# 2. Multiprocessing for CPU bound tasks
def cpu_bound_task(n):
    return sum(i * i for i in range(n))

with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(cpu_bound_task, [100000] * 4))

# 3. Thread-safe operations
import queue
thread_safe_queue = queue.Queue()
lock = threading.Lock()

def thread_safe_function():
    with lock:
        # Critical section
        pass
```

---

## Block 4: OOP & Design Patterns (30 minutes)

### Classes and Inheritance (Asked 85% of the time)
```python
# 1. Property decorators
class Circle:
    def __init__(self, radius):
        self._radius = radius
    
    @property
    def radius(self):
        return self._radius
    
    @radius.setter
    def radius(self, value):
        if value < 0:
            raise ValueError("Radius cannot be negative")
        self._radius = value
    
    @property
    def area(self):
        return 3.14159 * self._radius ** 2

# 2. Abstract Base Classes
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass
    
    @abstractmethod
    def perimeter(self):
        pass

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height
    
    def perimeter(self):
        return 2 * (self.width + self.height)

# 3. Multiple inheritance and MRO
class A:
    def method(self):
        print("A")

class B(A):
    def method(self):
        print("B")
        super().method()

class C(A):
    def method(self):
        print("C")
        super().method()

class D(B, C):
    def method(self):
        print("D")
        super().method()

# D().method() prints: D, B, C, A
```

### Design Patterns (Asked 60% of the time)
```python
# 1. Singleton Pattern
class Singleton:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

# 2. Factory Pattern
class AnimalFactory:
    @staticmethod
    def create_animal(animal_type):
        if animal_type == "dog":
            return Dog()
        elif animal_type == "cat":
            return Cat()
        else:
            raise ValueError(f"Unknown animal type: {animal_type}")

# 3. Observer Pattern
class Observable:
    def __init__(self):
        self._observers = []
    
    def attach(self, observer):
        self._observers.append(observer)
    
    def notify(self, message):
        for observer in self._observers:
            observer.update(message)

class Observer:
    def update(self, message):
        print(f"Observer received: {message}")
```

---

## Block 5: Error Handling & Exceptions (30 minutes)

```python
# 1. Custom exceptions (commonly asked)
class ValidationError(Exception):
    def __init__(self, message, field_name):
        super().__init__(message)
        self.field_name = field_name

class APIError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        super().__init__(f"API Error {status_code}: {message}")

# 2. Exception chaining
def process_data(data):
    try:
        # Some processing
        result = risky_operation(data)
    except ValueError as e:
        raise ProcessingError("Failed to process data") from e

# 3. Exception handling best practices
try:
    # Risky operation
    result = dangerous_function()
except SpecificError as e:
    # Handle specific error
    logging.error(f"Specific error occurred: {e}")
    raise  # Re-raise if cannot handle
except Exception as e:
    # Handle general errors
    logging.error(f"Unexpected error: {e}")
    raise  # Always re-raise unexpected errors
finally:
    # Cleanup code
    cleanup_resources()
```

---

## Block 6: Backend Specifics (45 minutes)

### Decorators (Asked 95% of the time)
```python
import functools
import time

# 1. Basic decorator
def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.2f} seconds")
        return result
    return wrapper

# 2. Decorator with parameters
def retry(max_attempts=3):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise e
                    time.sleep(1)
        return wrapper
    return decorator

# 3. Class-based decorator
class RateLimiter:
    def __init__(self, max_calls=10, period=60):
        self.max_calls = max_calls
        self.period = period
        self.calls = []
    
    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            # Remove calls older than period
            self.calls = [call_time for call_time in self.calls if now - call_time < self.period]
            
            if len(self.calls) >= self.max_calls:
                raise Exception("Rate limit exceeded")
            
            self.calls.append(now)
            return func(*args, **kwargs)
        return wrapper
```

### FastAPI Patterns (Critical for the role)
```python
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List
import asyncio

app = FastAPI()

# 1. Pydantic models
class DocumentRequest(BaseModel):
    title: str
    content: str
    metadata: Optional[dict] = None

class DocumentResponse(BaseModel):
    id: int
    title: str
    status: str
    processed_at: Optional[str] = None

# 2. Dependency injection
async def get_database():
    # Simulate database connection
    return {"connection": "active"}

# 3. Background tasks
def process_document_background(document_id: int):
    # Simulate document processing
    print(f"Processing document {document_id}")
    time.sleep(2)
    print(f"Document {document_id} processed")

# 4. API endpoints
@app.post("/documents/", response_model=DocumentResponse)
async def create_document(
    document: DocumentRequest,
    background_tasks: BackgroundTasks,
    db = Depends(get_database)
):
    # Create document
    doc_id = 123
    
    # Add background processing
    background_tasks.add_task(process_document_background, doc_id)
    
    return DocumentResponse(
        id=doc_id,
        title=document.title,
        status="processing"
    )

@app.get("/documents/{document_id}")
async def get_document(document_id: int, db = Depends(get_database)):
    if document_id not in [123, 456]:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return DocumentResponse(
        id=document_id,
        title="Sample Document",
        status="completed"
    )

# 5. Middleware
@app.middleware("http")
async def add_process_time_header(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response
```

### Database Connection Patterns
```python
# 1. PostgreSQL with asyncpg
import asyncpg
import asyncio

class PostgreSQLManager:
    def __init__(self, database_url):
        self.database_url = database_url
        self.pool = None
    
    async def create_pool(self):
        self.pool = await asyncpg.create_pool(self.database_url)
    
    async def execute_query(self, query, *args):
        async with self.pool.acquire() as connection:
            return await connection.fetch(query, *args)
    
    async def close_pool(self):
        await self.pool.close()

# Usage
async def database_example():
    db = PostgreSQLManager("postgresql://user:pass@localhost/db")
    await db.create_pool()
    
    results = await db.execute_query(
        "SELECT * FROM documents WHERE status = $1", "processed"
    )
    
    await db.close_pool()

# 2. Redis with aioredis
import aioredis

class RedisManager:
    def __init__(self, redis_url):
        self.redis_url = redis_url
        self.redis = None
    
    async def connect(self):
        self.redis = await aioredis.from_url(self.redis_url)
    
    async def set_cache(self, key, value, expire=3600):
        await self.redis.set(key, value, ex=expire)
    
    async def get_cache(self, key):
        return await self.redis.get(key)
    
    async def close(self):
        await self.redis.close()

# 3. SQLAlchemy async patterns
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

class SQLAlchemyManager:
    def __init__(self, database_url):
        self.engine = create_async_engine(database_url)
        self.async_session = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )
    
    async def get_session(self):
        async with self.async_session() as session:
            yield session
```

### Memory Management & Performance
```python
# 1. __slots__ for memory optimization (asked 40% of the time)
class OptimizedClass:
    __slots__ = ['x', 'y', 'z']
    
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

# 2. Weak references
import weakref

class Parent:
    def __init__(self):
        self.children = []
    
    def add_child(self, child):
        self.children.append(weakref.ref(child))

# 3. Generator for memory efficiency
def fibonacci_generator(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

# 4. Caching with functools
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_function(n):
    # Simulate expensive computation
    time.sleep(1)
    return n ** 2
```

---

## Practice Problems (30 minutes)

### Problem 1: Implement a LRU Cache
```python
class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.order = []
    
    def get(self, key: int) -> int:
        if key in self.cache:
            # Move to end (most recently used)
            self.order.remove(key)
            self.order.append(key)
            return self.cache[key]
        return -1
    
    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            # Update existing
            self.cache[key] = value
            self.order.remove(key)
            self.order.append(key)
        else:
            # Add new
            if len(self.cache) >= self.capacity:
                # Remove least recently used
                oldest = self.order.pop(0)
                del self.cache[oldest]
            
            self.cache[key] = value
            self.order.append(key)

# Test
cache = LRUCache(2)
cache.put(1, 1)
cache.put(2, 2)
print(cache.get(1))  # 1
cache.put(3, 3)      # evicts key 2
print(cache.get(2))  # -1 (not found)
```

### Problem 2: Async Rate Limiter
```python
import asyncio
import time
from collections import defaultdict

class AsyncRateLimiter:
    def __init__(self, max_requests: int, time_window: int):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = defaultdict(list)
        self.lock = asyncio.Lock()
    
    async def is_allowed(self, user_id: str) -> bool:
        async with self.lock:
            now = time.time()
            user_requests = self.requests[user_id]
            
            # Remove old requests
            self.requests[user_id] = [
                req_time for req_time in user_requests 
                if now - req_time < self.time_window
            ]
            
            # Check if under limit
            if len(self.requests[user_id]) < self.max_requests:
                self.requests[user_id].append(now)
                return True
            
            return False

# Test
async def test_rate_limiter():
    limiter = AsyncRateLimiter(max_requests=3, time_window=60)
    
    for i in range(5):
        allowed = await limiter.is_allowed("user1")
        print(f"Request {i+1}: {'Allowed' if allowed else 'Blocked'}")
```

### Problem 3: Document Processing Pipeline
```python
import asyncio
from enum import Enum
from typing import List, Dict, Any

class ProcessingStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class Document:
    def __init__(self, doc_id: str, content: str):
        self.id = doc_id
        self.content = content
        self.status = ProcessingStatus.PENDING
        self.metadata = {}

class DocumentProcessor:
    def __init__(self):
        self.processing_queue = asyncio.Queue()
        self.results = {}
    
    async def submit_document(self, document: Document):
        await self.processing_queue.put(document)
        return document.id
    
    async def process_document(self, document: Document) -> Dict[str, Any]:
        # Simulate processing
        document.status = ProcessingStatus.PROCESSING
        await asyncio.sleep(1)  # Simulate work
        
        # Extract metadata
        word_count = len(document.content.split())
        char_count = len(document.content)
        
        document.metadata = {
            "word_count": word_count,
            "char_count": char_count,
            "processed_at": time.time()
        }
        
        document.status = ProcessingStatus.COMPLETED
        return document.metadata
    
    async def worker(self):
        while True:
            try:
                document = await self.processing_queue.get()
                result = await self.process_document(document)
                self.results[document.id] = result
                self.processing_queue.task_done()
            except Exception as e:
                print(f"Error processing document: {e}")
    
    async def start_workers(self, num_workers: int = 3):
        workers = [asyncio.create_task(self.worker()) for _ in range(num_workers)]
        return workers

# Test
async def test_document_processor():
    processor = DocumentProcessor()
    workers = await processor.start_workers(3)
    
    # Submit documents
    documents = [
        Document("doc1", "This is a sample document"),
        Document("doc2", "Another document with more content here"),
        Document("doc3", "Short doc")
    ]
    
    for doc in documents:
        await processor.submit_document(doc)
    
    # Wait for processing
    await processor.processing_queue.join()
    
    # Cancel workers
    for worker in workers:
        worker.cancel()
    
    print("Results:", processor.results)
```

---

## Quick Reference Cheatsheet

### Most Asked Topics (Frequency)
1. **async/await** (95% of backend interviews)
2. **Decorators** (95%)
3. **List comprehensions vs generators** (90%)
4. **Property decorators** (85%)
5. **String formatting** (80%)
6. **Context managers** (75%)
7. **Threading vs multiprocessing** (70%)
8. **Exception handling** (70%)
9. **Design patterns** (60%)
10. **Memory optimization** (40%)

### Key Points to Mention
- **Memory efficiency**: Use generators for large datasets
- **Concurrency**: async for I/O-bound, multiprocessing for CPU-bound
- **Error handling**: Always use specific exceptions
- **Resource management**: Always use context managers
- **Performance**: Profile before optimizing

---

See also: [Python Implementation](python_implementation.md), [System Architecture & Design](system_architecture.md)

For guidance on how to use and update this material, see the [living guide note](README.md#living-guide-note). 