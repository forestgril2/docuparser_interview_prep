# Python 1-Hour Focus: Critical Interview Topics

**Time Budget: 60 minutes**
- 20 min: Async Programming & Decorators (95% interview frequency)
- 15 min: OOP & Design Patterns  
- 10 min: FastAPI Patterns (Critical for this role)
- 10 min: Quick Practice Problem
- 5 min: Cheat Sheet Review

---

## 🔥 Block 1: Async Programming & Decorators (20 minutes)

### async/await (Asked in 95% of backend interviews)
```python
import asyncio
import aiohttp

# Basic async function
async def fetch_data(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

# Running multiple tasks concurrently
async def fetch_multiple():
    urls = ['http://api1.com', 'http://api2.com']
    tasks = [fetch_data(url) for url in urls]
    results = await asyncio.gather(*tasks)
    return results

# Async context manager (commonly asked)
class AsyncDBConnection:
    async def __aenter__(self):
        self.conn = await connect_to_db()
        return self.conn
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.conn.close()
        return False

# Usage
async def process_data():
    async with AsyncDBConnection() as db:
        return await db.execute("SELECT * FROM documents")
```

### Decorators (Asked in 95% of backend interviews)
```python
import functools
import time

# Basic decorator with timer
def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__}: {time.time() - start:.2f}s")
        return result
    return wrapper

# Decorator with parameters
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

# Class-based decorator
class RateLimiter:
    def __init__(self, max_calls=10, period=60):
        self.max_calls = max_calls
        self.period = period
        self.calls = []
    
    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            self.calls = [t for t in self.calls if now - t < self.period]
            
            if len(self.calls) >= self.max_calls:
                raise Exception("Rate limit exceeded")
            
            self.calls.append(now)
            return func(*args, **kwargs)
        return wrapper

# Usage examples
@timer
@retry(max_attempts=3)
def api_call():
    # Simulate API call
    pass

@RateLimiter(max_calls=5, period=60)
def limited_function():
    pass
```

---

## 🎯 Block 2: OOP & Design Patterns (15 minutes)

### Property Decorators (Asked in 85% of interviews)
```python
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

# Usage
c = Circle(5)
print(c.area)  # 78.54
c.radius = 10  # Uses setter validation
```

### Abstract Base Classes
```python
from abc import ABC, abstractmethod

class DocumentProcessor(ABC):
    @abstractmethod
    def process(self, document):
        pass
    
    @abstractmethod
    def validate(self, document):
        pass

class PDFProcessor(DocumentProcessor):
    def process(self, document):
        return f"Processing PDF: {document}"
    
    def validate(self, document):
        return document.endswith('.pdf')
```

### Essential Design Patterns
```python
# Singleton (thread-safe)
import threading

class DatabaseConfig:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

# Factory Pattern
class DocumentFactory:
    @staticmethod
    def create_processor(doc_type):
        processors = {
            'pdf': PDFProcessor,
            'docx': DocxProcessor,
            'txt': TextProcessor
        }
        processor_class = processors.get(doc_type)
        if not processor_class:
            raise ValueError(f"Unknown document type: {doc_type}")
        return processor_class()
```

---

## 🚀 Block 3: FastAPI Patterns (10 minutes)

```python
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

# Pydantic models
class DocumentRequest(BaseModel):
    title: str
    content: str
    metadata: Optional[dict] = None

class DocumentResponse(BaseModel):
    id: int
    title: str
    status: str

# Dependency injection
async def get_db():
    return {"connection": "active"}

# Background tasks
def process_document_bg(doc_id: int):
    print(f"Processing document {doc_id}")

# API endpoints
@app.post("/documents/", response_model=DocumentResponse)
async def create_document(
    document: DocumentRequest,
    background_tasks: BackgroundTasks,
    db = Depends(get_db)
):
    doc_id = 123
    background_tasks.add_task(process_document_bg, doc_id)
    
    return DocumentResponse(
        id=doc_id,
        title=document.title,
        status="processing"
    )

@app.get("/documents/{document_id}")
async def get_document(document_id: int):
    if document_id not in [123, 456]:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return DocumentResponse(
        id=document_id,
        title="Sample Document",
        status="completed"
    )

# Middleware
@app.middleware("http")
async def add_process_time(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response
```

---

## 💡 Block 4: Quick Practice Problem (10 minutes)

### Document Processing Pipeline
```python
import asyncio
from enum import Enum
from typing import Dict, Any

class ProcessingStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class DocumentProcessor:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.results = {}
    
    async def submit_document(self, doc_id: str, content: str):
        await self.queue.put({"id": doc_id, "content": content})
        return doc_id
    
    async def process_document(self, document: Dict[str, Any]):
        # Simulate processing
        await asyncio.sleep(0.1)
        
        word_count = len(document["content"].split())
        return {
            "word_count": word_count,
            "char_count": len(document["content"]),
            "status": ProcessingStatus.COMPLETED
        }
    
    async def worker(self):
        while True:
            try:
                document = await self.queue.get()
                result = await self.process_document(document)
                self.results[document["id"]] = result
                self.queue.task_done()
            except Exception as e:
                print(f"Error: {e}")
    
    async def start_processing(self, num_workers=3):
        workers = [asyncio.create_task(self.worker()) for _ in range(num_workers)]
        return workers

# Test the processor
async def test_processor():
    processor = DocumentProcessor()
    workers = await processor.start_processing(3)
    
    # Submit documents
    await processor.submit_document("doc1", "This is test content")
    await processor.submit_document("doc2", "Another document with more words here")
    
    # Wait for processing
    await processor.queue.join()
    
    # Cancel workers
    for worker in workers:
        worker.cancel()
    
    print("Results:", processor.results)

# Run: asyncio.run(test_processor())
```

---

## 📋 Quick Reference Cheat Sheet (5 minutes)

### Most Asked Topics (Priority Order)
1. **async/await** (95%) - Master this completely
2. **Decorators** (95%) - Know all 3 types
3. **List comprehensions vs generators** (90%)
4. **Property decorators** (85%)
5. **String formatting** (80%)
6. **Context managers** (75%)
7. **Threading vs multiprocessing** (70%)

### Key Performance Tips
```python
# Memory efficiency
large_gen = (x**2 for x in range(1000000))  # Generator
small_list = [x**2 for x in range(100)]     # List for small data

# Caching
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_function(n):
    return n ** 2

# String operations
words = ['hello', 'world']
efficient = ' '.join(words)    # Fast
slow = words[0] + ' ' + words[1]  # Avoid for many strings
```

### Exception Handling
```python
# Custom exceptions
class ValidationError(Exception):
    def __init__(self, message, field):
        super().__init__(message)
        self.field = field

# Exception chaining
try:
    risky_operation()
except ValueError as e:
    raise ProcessingError("Processing failed") from e
```

### Database Patterns
```python
# Async PostgreSQL
import asyncpg

async def get_data():
    conn = await asyncpg.connect("postgresql://...")
    try:
        return await conn.fetch("SELECT * FROM documents")
    finally:
        await conn.close()

# With context manager
@contextmanager
async def db_connection():
    conn = await asyncpg.connect("postgresql://...")
    try:
        yield conn
    finally:
        await conn.close()
```

### Interview Success Formula
1. **Start simple** → Add complexity gradually
2. **Explain trade-offs** → Memory vs speed, sync vs async
3. **Handle errors** → Always consider edge cases
4. **Use context managers** → For any resource cleanup
5. **Think async** → For I/O operations

### Common Mistakes to Avoid
- ❌ Using `list()` on large generators
- ❌ Forgetting `@functools.wraps` in decorators
- ❌ Not handling exceptions in async code
- ❌ Using `__del__` for critical cleanup
- ❌ Mixing sync and async code incorrectly

---

## Final Tips for Your Interview
1. **Mention async patterns** when discussing I/O operations
2. **Use context managers** for any resource management
3. **Explain memory efficiency** when choosing data structures
4. **Show decorator knowledge** for cross-cutting concerns
5. **Demonstrate FastAPI familiarity** for the Docparser role

**Remember:** If asked about document processing, combine async programming + context managers + background tasks for a complete answer!

---

See also: [Python Implementation](python_implementation.md), [System Architecture & Design](system_architecture.md)

For guidance on how to use and update this material, see the [living guide note](README.md#living-guide-note). 