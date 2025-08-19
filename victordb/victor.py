from __future__ import annotations
import socket
import struct
import cbor2
import json
import pickle
from dataclasses import dataclass, field, asdict
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar
from enum import IntEnum
from snowflake import SnowflakeGenerator

class MessageType(IntEnum):
    # Vector protocol message types
    MSG_INSERT          = 0x01
    MSG_DELETE          = 0x02
    MSG_SEARCH          = 0x03
    MSG_MATCH_RESULT    = 0x04


    # Key-Value protocol message types
    MSG_PUT             = 0x06
    MSG_DEL             = 0x07
    MSG_GET             = 0x08
    MSG_GET_RESULT      = 0x09

    MSG_OP_RESULT       = 0x0A
    MSG_ERROR           = 0x0B


class VictorError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


class VictorClientBase:
    """
    Base client class for VictorDB server communication.
    
    Handles low-level socket operations, message serialization/deserialization,
    and connection management. This class should be inherited by specific
    client implementations for vector index and table operations.
    """
    
    def __init__(self):
        self.sock: Optional[socket.socket] = None

    def connect(self, *, host: Optional[str] = None, port: Optional[int] = None, unix_path: Optional[str] = None):
        """
        Connects to the server via TCP or Unix socket.
        
        Args:
            host: TCP hostname or IP address
            port: TCP port number
            unix_path: Unix domain socket path
            
        Raises:
            ValueError: If neither host/port nor unix_path is provided
            ConnectionError: If connection fails
        """
        if unix_path:
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.sock.connect(unix_path)
        elif host and port:
            self.sock = socket.create_connection((host, port))
        else:
            raise ValueError("Must provide either host/port or unix_path")

    def close(self):
        """Close the socket connection."""
        if self.sock:
            self.sock.close()
            self.sock = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _send_msg(self, msg_type: int, payload: bytes):
        """
        Send a message to the server.
        
        Args:
            msg_type: Message type identifier
            payload: CBOR-encoded message payload
            
        Raises:
            ConnectionError: If socket is not connected
            ValueError: If message type or payload is invalid
        """
        if not self.sock:
            raise ConnectionError("Socket is not connected")

        if msg_type > 0xF or len(payload) > 0x0FFFFFFF:
            raise ValueError("Invalid message type or payload too large")

        header = ((msg_type & 0xF) << 28) | (len(payload) & 0x0FFFFFFF)
        raw = struct.pack("!I", header)
        self.sock.sendall(raw + payload)

    def _recv_msg(self) -> Tuple[int, bytes]:
        """
        Receive a message from the server.
        
        Returns:
            Tuple of (message_type, payload)
            
        Raises:
            VictorError: If server returns an error message
            ConnectionError: If socket connection is lost
        """
        hdr_raw = self._recv_all(4)
        hdr_val = struct.unpack("!I", hdr_raw)[0]
        msg_type = (hdr_val >> 28) & 0xF
        msg_len = hdr_val & 0x0FFFFFFF
        payload = self._recv_all(msg_len)

        if msg_type == MessageType.MSG_ERROR:
            code, msg = cbor2.loads(payload)
            raise VictorError(code, msg)

        return msg_type, payload

    def _recv_all(self, n: int) -> bytes:
        """
        Receive exactly n bytes from the socket.
        
        Args:
            n: Number of bytes to receive
            
        Returns:
            Received bytes
            
        Raises:
            ConnectionError: If socket is closed before receiving all bytes
        """
        buf = bytearray()
        while len(buf) < n:
            if self.sock is None:
                raise ValueError("Not connected")
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("Socket closed")
            buf.extend(chunk)
        return bytes(buf)


class VictorIndexClient(VictorClientBase):
    """
    Client for VictorDB vector index operations.
    
    Provides high-level interface for vector operations including INSERT,
    SEARCH, and DELETE on the vector index server.
    """

    def insert(self, id: int, vector: List[float]) -> int:
        """
        Insert a vector into the index.
        
        Args:
            id: Unique identifier for the vector
            vector: Vector data as list of floats
            
        Returns:
            The ID of the inserted vector
            
        Raises:
            VictorError: If insertion fails
        """
        msg = cbor2.dumps([id, vector])
        self._send_msg(MessageType.MSG_INSERT, msg)
        msg_type, payload = self._recv_msg()
        if msg_type != MessageType.MSG_OP_RESULT:
            raise VictorError(-1, f"Unexpected message type {msg_type}, expected INSERT_RESULT")
        [code, message] = cbor2.loads(payload)
        if code != 0:  # SUCCESS = 0
            raise VictorError(code, message)
        return id

    def delete(self, id_: int) -> bool:
        """
        Delete a vector from the index.
        
        Args:
            id_: ID of the vector to delete
            
        Returns:
            True if deletion was successful
            
        Raises:
            VictorError: If deletion fails
        """
        msg = cbor2.dumps([id_])
        self._send_msg(MessageType.MSG_DELETE, msg)
        msg_type, payload = self._recv_msg()
        if msg_type != MessageType.MSG_OP_RESULT:
            raise VictorError(-1, f"Unexpected message type {msg_type}, expected DELETE_RESULT")
        [code, message] = cbor2.loads(payload)
        if code != 0:
            raise VictorError(code, message)
        return code == 0  # SUCCESS = 0

    def search(self, vector: List[float], topk: int) -> List[Tuple[int, float]]:
        """
        Search for similar vectors in the index.
        
        Args:
            vector: Query vector as list of floats
            topk: Number of top results to return
            
        Returns:
            List of (id, distance) tuples for the closest vectors
            
        Raises:
            VictorError: If search fails
        """
        msg = cbor2.dumps([vector, topk])
        self._send_msg(MessageType.MSG_SEARCH, msg)
        msg_type, payload = self._recv_msg()
        
        if msg_type != MessageType.MSG_MATCH_RESULT:
            raise VictorError(-1, f"Unexpected message type {msg_type}, expected MATCH_RESULT")
        
        results = cbor2.loads(payload)
        return [(int(id_), float(distance)) for id_, distance in results]


class VictorTableClient(VictorClientBase):
    """
    Client for VictorDB table (key-value) operations.
    
    Provides high-level interface for key-value operations including PUT,
    GET, and DEL on the table server.
    """

    @staticmethod
    def to_bytes(value: Any) -> bytes:
        """
        Convert any Python value to bytes for storage.
        
        Args:
            value: Any Python value to convert
            
        Returns:
            Bytes representation of the value
            
        Examples:
            >>> VictorTableClient.to_bytes("hello")
            b'hello'
            >>> VictorTableClient.to_bytes({"name": "Alice", "age": 30})
            b'{"name": "Alice", "age": 30}'
            >>> VictorTableClient.to_bytes(42)
            b'42'
        """
        if isinstance(value, bytes):
            return value
        elif isinstance(value, str):
            return value.encode('utf-8')
        elif isinstance(value, (int, float)):
            return str(value).encode('utf-8')
        elif isinstance(value, (dict, list, tuple)):
            return json.dumps(value, ensure_ascii=False).encode('utf-8')
        elif hasattr(value, '__dict__'):
            # Objects with attributes - convert to dict first
            return json.dumps(value.__dict__, ensure_ascii=False).encode('utf-8')
        else:
            # Fallback to pickle for complex objects
            return pickle.dumps(value)

    @staticmethod
    def from_bytes(data: bytes, target_type: str = 'auto') -> Any:
        """
        Convert bytes back to Python values.
        
        Args:
            data: Bytes to convert
            target_type: Target type ('auto', 'str', 'json', 'int', 'float', 'pickle')
            
        Returns:
            Converted Python value
            
        Examples:
            >>> VictorTableClient.from_bytes(b'hello')
            'hello'
            >>> VictorTableClient.from_bytes(b'{"name": "Alice"}', 'json')
            {'name': 'Alice'}
            >>> VictorTableClient.from_bytes(b'42', 'int')
            42
        """
        if target_type == 'str' or target_type == 'auto':
            try:
                return data.decode('utf-8')
            except UnicodeDecodeError:
                if target_type == 'str':
                    raise
                # Fall through to pickle for auto mode
        
        if target_type == 'json':
            return json.loads(data.decode('utf-8'))
        elif target_type == 'int':
            return int(data.decode('utf-8'))
        elif target_type == 'float':
            return float(data.decode('utf-8'))
        elif target_type == 'pickle':
            return pickle.loads(data)
        elif target_type == 'auto':
            # Try JSON first, then pickle as fallback
            try:
                text = data.decode('utf-8')
                # Try to parse as JSON
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    # Return as string if not JSON
                    return text
            except UnicodeDecodeError:
                # Must be pickled data
                return pickle.loads(data)
        else:
            raise ValueError(f"Unknown target_type: {target_type}")

    def put(self, key: bytes, value: bytes) -> bool:
        """
        Store a key-value pair in the table.
        
        Args:
            key: Key as bytes
            value: Value as bytes
            
        Returns:
            True if operation was successful
            
        Raises:
            VictorError: If put operation fails
        """
        msg = cbor2.dumps([key, value])
        self._send_msg(MessageType.MSG_PUT, msg)
        msg_type, payload = self._recv_msg()
        if msg_type != MessageType.MSG_OP_RESULT:
            raise VictorError(-1, f"Unexpected message type {msg_type}, expected PUT_RESULT")
        [code, message] = cbor2.loads(payload)
        if code != 0:
            raise VictorError(code, message)
        return True

    def get(self, key: bytes) -> bytes | None:
        """
        Retrieve a value by key from the table.
        
        Args:
            key: Key as bytes
            
        Returns:
            Value as bytes
            
        Raises:
            VictorError: If key is not found or operation fails
        """
        msg = cbor2.dumps([key])
        self._send_msg(MessageType.MSG_GET, msg)
        try:
            msg_type, payload = self._recv_msg()
        except VictorError as e:
            if e.code == 1:
                return None
            else:
                raise e
        if msg_type != MessageType.MSG_GET_RESULT:
            raise VictorError(-1, f"Unexpected message type {msg_type}, expected GET_RESULT")
        [value] = cbor2.loads(payload)
        return value

    def delete(self, key: bytes) -> bool:
        """
        Delete a key-value pair from the table.
        
        Args:
            key: Key as bytes
            
        Returns:
            True if deletion was successful
            
        Raises:
            VictorError: If deletion fails
        """
        msg = cbor2.dumps([key])
        self._send_msg(MessageType.MSG_DEL, msg)
        msg_type, payload = self._recv_msg()
        if msg_type != MessageType.MSG_OP_RESULT:
            raise VictorError(-1, f"Unexpected message type {msg_type}, expected DEL_RESULT")

        [code, message] = cbor2.loads(payload)
        if code != 0:
            raise VictorError(code, message)
        return True  # SUCCESS = 0


T = TypeVar("T", bound="VictorBaseModel")

class VictorSession(object):
    """
    Abstracts access to VictorTableClient and provides keyspace utilities.
    """
    def __init__(self, table_client, snowflake_node_id: int = 42, debug: bool = False):
        self.table = table_client
        self._id_gen = SnowflakeGenerator(snowflake_node_id)
        self.debug = debug

    def _debug_print(self, *args, **kwargs):
        """Print debug messages only if debug mode is enabled."""
        if self.debug:
            print(*args, **kwargs)

    def new_id(self) -> int:
        val = next(self._id_gen)
        if val is None:
            raise ValueError("SnowflakeGenerator returned None, expected int")
        return val

    # Key conventions:
    #  - Record:     {Class}:{id}
    #  - All list:   {Class}:_all        -> list[int]  (ids)
    #  - Eq index:   idx:{Class}:{field}:{hash} -> list[int] (ids)
    #  - Shadow for simple concurrency: optional (not implemented here)

    @staticmethod
    def key_for_record(cls_name: str, sid: int) -> str:
        return f"{cls_name}:{sid}"

    @staticmethod
    def key_for_all(cls_name: str) -> str:
        return f"{cls_name}:_all"

    @staticmethod
    def key_for_index_eq(cls_name: str, field: str, value: Any) -> str:
        hv = VictorSession._hash_value(value)
        return f"idx:{cls_name}:{field}:{hv}"

    @staticmethod
    def _hash_value(value: Any) -> str:
        # Stable and compact hash for values in indexes.
        # For long strings avoids giant keys.
        import hashlib
        # Normalize to JSON so that 1 == 1.0 and dicts/lists are stable
        s = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.blake2b(s.encode("utf-8"), digest_size=10).hexdigest()

    # KV helpers with (de)serialization
    def kv_put(self, key: str, value: Any) -> None:
        self.table.put(self.table.to_bytes(key), self.table.to_bytes(value))

    def kv_get(self, key: str, target_type='auto') -> Optional[Any]:
        raw = self.table.get(self.table.to_bytes(key))
        if raw is None:
            return None
        return self.table.from_bytes(raw, target_type)

    def kv_del(self, key: str) -> bool:
        return self.table.delete(key)

    # List structures (ids) for _all and indexes
    def _list_add(self, key: str, id_: int) -> None:
        self._debug_print(f"DEBUG _list_add: key={key}, id={id_}")
        cur = self.kv_get(key, 'json')
        self._debug_print(f"DEBUG _list_add: current value={cur}")
        if cur is None or not isinstance(cur, list):
            cur = [id_]
        else:
            if id_ not in cur:
                cur.append(id_)
        self._debug_print(f"DEBUG _list_add: saving value={cur}")
        self.kv_put(key, cur)

    def _list_remove(self, key: str, id_: int) -> None:
        cur = self.kv_get(key, 'json')
        if not cur or not isinstance(cur, list):
            return
        try:
            cur.remove(id_)
        except ValueError:
            pass
        self.kv_put(key, cur)


# -----------------------
# Victor ORM Base
# -----------------------

@dataclass
class VictorBaseModel:
    """
    Base model:
    - PK `id` (snowflake)
    - Serialization conventions
    - Save / delete / get / query_eq
    - Configurable equality indexes in `__indexed__`
    """
    id: Optional[int] = field(default=None, compare=True)

    # Class config
    __classname__: ClassVar[str] = "VictorBaseModel"
    __indexed__: ClassVar[List[str]] = []  # fields indexed by equality

    # ---- keys
    @classmethod
    def _record_key(cls, sid: int) -> str:
        return VictorSession.key_for_record(cls.__classname__, sid)

    @classmethod
    def _all_key(cls) -> str:
        return VictorSession.key_for_all(cls.__classname__)

    @classmethod
    def _index_key(cls, field: str, value: Any) -> str:
        return VictorSession.key_for_index_eq(cls.__classname__, field, value)

    # ---- serialization
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        return cls(**data)  # dataclass reconstructs by fields

    # ---- CRUD
    def save(self: T, session: VictorSession) -> T:
        """
        Inserts or updates the record and maintains secondary indexes.
        If update: repairs indexes if an indexed field changed.
        """
        is_insert = self.id is None
        old: Optional[Dict[str, Any]] = None

        if is_insert:
            self.id = session.new_id()
            session._debug_print(f"DEBUG save: INSERT new record with id={self.id}")
        else:
            # load old version to see if indexes change
            if self.id is None:
                raise ValueError("Cannot update record: id is None")
            old_raw = session.kv_get(self._record_key(self.id), 'json')
            if isinstance(old_raw, dict):
                old = old_raw
            session._debug_print(f"DEBUG save: UPDATE record id={self.id}, old_data={old}")

        # Persist record
        record_data = self.to_dict()
        session._debug_print(f"DEBUG save: Persisting record_key={self._record_key(self.id)}, data={record_data}")
        session.kv_put(self._record_key(self.id), record_data)

        # Maintain list of all IDs
        session._debug_print(f"DEBUG save: Adding to all_key={self._all_key()}")
        session._list_add(self._all_key(), self.id)

        # Indexes: if insert → add; if update → remove old and add new
        if self.__indexed__:
            session._debug_print(f"DEBUG save: Processing indexes for fields: {self.__indexed__}")
            if old:
                for f in self.__indexed__:
                    old_v = old.get(f, None)
                    new_v = getattr(self, f, None)
                    session._debug_print(f"DEBUG save: INDEX UPDATE field={f}, old_value={old_v}, new_value={new_v}")
                    if old_v != new_v:
                        # remove from old index
                        if old_v is not None:
                            old_index_key = self._index_key(f, old_v)
                            session._debug_print(f"DEBUG save: Removing from old index: {old_index_key}")
                            session._list_remove(old_index_key, self.id)
                        # add to new index
                        if new_v is not None:
                            new_index_key = self._index_key(f, new_v)
                            session._debug_print(f"DEBUG save: Adding to new index: {new_index_key}")
                            session._list_add(new_index_key, self.id)
            else:
                # insert
                session._debug_print("DEBUG save: INSERT - adding to all indexes")
                for f in self.__indexed__:
                    v = getattr(self, f, None)
                    if v is not None:
                        index_key = self._index_key(f, v)
                        session._debug_print(f"DEBUG save: Adding to index field={f}, value={v}, key={index_key}")
                        session._list_add(index_key, self.id)

        return self

    def delete(self, session: VictorSession) -> None:
        if self.id is None:
            return
        # delete record
        session.kv_del(self._record_key(self.id))
        # remove from _all
        session._list_remove(self._all_key(), self.id)
        # remove from indexes
        if self.__indexed__:
            for f in self.__indexed__:
                v = getattr(self, f, None)
                if v is not None:
                    session._list_remove(self._index_key(f, v), self.id)

    def refresh(self: T, session: VictorSession) -> T:
        if self.id is None:
            raise ValueError("No ID to refresh")
        raw = session.kv_get(self._record_key(self.id))
        if not raw:
            raise KeyError("Record not found")
        obj = self.from_dict(raw)
        # copy fields
        for k, v in obj.__dict__.items():
            setattr(self, k, v)
        return self

    # ---- static reads
    @classmethod
    def get(cls: Type[T], session: VictorSession, id_: int) -> Optional[T]:
        raw = session.kv_get(cls._record_key(id_), 'json')
        if not raw:
            return None
        return cls.from_dict(raw)

    @classmethod
    def all_ids(cls, session: VictorSession) -> List[int]:
        ids = session.kv_get(cls._all_key(), 'json')
        return ids if isinstance(ids, list) else []

    @classmethod
    def query_eq(cls: Type[T], session: VictorSession, field: str, value: Any) -> List[T]:
        """
        Equality query. Requires that `field` is in __indexed__.
        Returns materialized objects (reads each record).
        """
        if field not in cls.__indexed__:
            raise ValueError(f"Field '{field}' is not indexed for class {cls.__classname__}")
        k = cls._index_key(field, value)
        session._debug_print(f"DEBUG query_eq: index_key={k}")
        ids = session.kv_get(k, 'json') or []
        session._debug_print(f"DEBUG query_eq: found_ids={ids}")
        if not isinstance(ids, list):
            session._debug_print(f"DEBUG query_eq: ids is not a list, got {type(ids)}: {ids}")
            return []
        out: List[T] = []
        for sid in ids:
            obj = cls.get(session, sid)
            session._debug_print(f"DEBUG query_eq: loaded object id={sid}: {obj}")
            if obj is not None:
                out.append(obj)
        return out

