# VictorDB Python Client

VictorDB is a Python client and ORM for high-performance vector and key-value databases. It provides a simple, flexible API for vector search, key-value storage, and object modeling, making it easy to build modern AI and data applications.

- Library Core (libvictor):  https://github.com/victor-base/libvictor
- Database (victordb): https://github.com/victor-base/victordb

## Features

- Vector index operations: insert, search, delete
- Key-value table operations: put, get, delete
- ORM-style data modeling with secondary indexes
- High-performance, binary protocol (CBOR)
- Pluggable and extensible design

## Installation

```bash
pip install victordb
```

## Quick Start

```python
from victordb.victor import VictorIndexClient, VictorTableClient, VictorSession, VictorBaseModel

# Connect to VictorDB server (vector index)
index_client = VictorIndexClient()
index_client.connect(host="localhost", port=9000)

# Insert a vector
index_client.insert(id=123, vector=[0.1, 0.2, 0.3])

# Search for similar vectors
results = index_client.search(vector=[0.1, 0.2, 0.3], topk=5)
print(results)

# Connect to VictorDB server (key-value table)
table_client = VictorTableClient()
table_client.connect(host="localhost", port=9001)

# Store and retrieve a value
table_client.put(b"mykey", b"myvalue")
value = table_client.get(b"mykey")
print(value)
```

## ORM Example

Define your own models by inheriting from `VictorBaseModel`:

```python
from victordb.victor import VictorSession, VictorTableClient, VictorBaseModel
from dataclasses import dataclass, field

@dataclass
class User(VictorBaseModel):
    __classname__ = "User"
    __indexed__ = ["email"]
    email: str = ""
    name: str = ""

# Connect to table and create session
table = VictorTableClient()
table.connect(host="localhost", port=9001)
session = VictorSession(table)

# Create and save a user
user = User(email="alice@example.com", name="Alice")
user.save(session)

# Query by indexed field
users = User.query_eq(session, "email", "alice@example.com")
print(users)
```

## API Overview

### VictorIndexClient

- `insert(id: int, vector: List[float]) -> int`
- `delete(id: int) -> bool`
- `search(vector: List[float], topk: int) -> List[Tuple[int, float]]`

### VictorTableClient

- `put(key: bytes, value: bytes) -> bool`
- `get(key: bytes) -> Optional[bytes]`
- `delete(key: bytes) -> bool`
- `to_bytes(value: Any) -> bytes`
- `from_bytes(data: bytes, target_type: str = 'auto') -> Any`

### VictorSession

- `new_id() -> int`
- `kv_put(key: str, value: Any) -> None`
- `kv_get(key: str, target_type='auto') -> Optional[Any]`

### VictorBaseModel

- `save(session: VictorSession) -> Self`
- `delete(session: VictorSession) -> None`
- `refresh(session: VictorSession) -> Self`
- `get(session: VictorSession, id_: int) -> Optional[Self]`
- `all_ids(session: VictorSession) -> List[int]`
- `query_eq(session: VictorSession, field: str, value: Any) -> List[Self]`

## License

MIT
