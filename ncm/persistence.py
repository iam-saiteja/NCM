"""NCM - Binary .ncm file persistence (v2 format)."""

import struct
import gzip
import io
import os
import time
import json

import numpy as np

from ncm.auto_state import AutoStateTracker
from ncm.memory import MemoryEntry, MemoryStore
from ncm.profile import MemoryProfile
from ncm.exceptions import PersistenceError, CorruptFileError

NCM_MAGIC = b'NCM\x02'
NCM_VERSION = 2

FLAG_COMPRESSED = 0b00000001
FLAG_HAS_PROFILE = 0b00000010
FLAG_FP16 = 0b00000100
FLAG_HAS_AUTOSTATE = 0b00001000


class NCMFile:
    """Reads and writes .ncm binary files."""

    @staticmethod
    def _read_exact(buf, nbytes: int, context: str) -> bytes:
        data = buf.read(nbytes)
        if len(data) != nbytes:
            raise CorruptFileError("<buffer>", f"Truncated data while reading {context}: expected {nbytes} bytes, got {len(data)}")
        return data

    @staticmethod
    def save(store: MemoryStore, path: str, compress: bool = True, fp16: bool = True) -> None:
        try:
            buf = io.BytesIO()
            flags = FLAG_HAS_PROFILE
            if compress:
                flags |= FLAG_COMPRESSED
            if fp16:
                flags |= FLAG_FP16
            flags |= FLAG_HAS_AUTOSTATE

            memories = store.get_all_safe()
            buf.write(NCM_MAGIC)
            buf.write(struct.pack('>B', NCM_VERSION))
            buf.write(struct.pack('>H', flags))
            buf.write(struct.pack('>Q', store.step))
            buf.write(struct.pack('>I', len(memories)))
            buf.write(struct.pack('>q', int(time.time())))

            profile_bytes = store.profile.to_json()
            buf.write(struct.pack('>I', len(profile_bytes)))
            buf.write(profile_bytes)

            auto_state_bytes = json.dumps(store.auto_state.to_dict()).encode("utf-8")
            buf.write(struct.pack('>I', len(auto_state_bytes)))
            buf.write(auto_state_bytes)

            mem_buf = io.BytesIO()
            sem_dim = store.profile.semantic_dim
            emo_dim = store.profile.emotional_dim
            state_dim = store.profile.state_dim
            
            for memory in memories:
                NCMFile._write_memory(
                    mem_buf,
                    memory,
                    sem_dim,
                    emo_dim,
                    state_dim,
                    fp16=fp16,
                    has_autostate=True,
                )

            mem_data = mem_buf.getvalue()
            if compress:
                mem_data = gzip.compress(mem_data)
            buf.write(mem_data)

            os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
            with open(path, 'wb') as f:
                f.write(buf.getvalue())
        except (OSError, struct.error) as e:
            raise PersistenceError('save', str(e))

    @staticmethod
    def load(path: str) -> MemoryStore:
        if not os.path.exists(path):
            return MemoryStore()
        try:
            with open(path, 'rb') as f:
                data = f.read()
            buf = io.BytesIO(data)

            magic = buf.read(4)
            if magic != NCM_MAGIC:
                raise CorruptFileError(path, f"Invalid magic bytes: {magic!r}")

            version = struct.unpack('>B', buf.read(1))[0]
            if version != NCM_VERSION:
                raise CorruptFileError(path, f"Unsupported version: {version}")

            flags = struct.unpack('>H', buf.read(2))[0]
            compressed = bool(flags & FLAG_COMPRESSED)
            has_profile = bool(flags & FLAG_HAS_PROFILE)
            fp16 = bool(flags & FLAG_FP16)
            has_autostate = bool(flags & FLAG_HAS_AUTOSTATE)

            step = struct.unpack('>Q', buf.read(8))[0]
            memory_count = struct.unpack('>I', buf.read(4))[0]
            created_at = struct.unpack('>q', buf.read(8))[0]

            profile = MemoryProfile()
            if has_profile:
                profile_len = struct.unpack('>I', buf.read(4))[0]
                profile_bytes = buf.read(profile_len)
                profile = MemoryProfile.from_json(profile_bytes)

            tracker = AutoStateTracker()
            if has_autostate:
                auto_state_len = struct.unpack('>I', buf.read(4))[0]
                auto_state_bytes = buf.read(auto_state_len)
                try:
                    tracker = AutoStateTracker.from_dict(json.loads(auto_state_bytes.decode("utf-8")))
                except Exception:
                    tracker = AutoStateTracker()

            mem_data = buf.read()
            if compressed:
                mem_data = gzip.decompress(mem_data)

            mem_buf = io.BytesIO(mem_data)
            store = MemoryStore(profile=profile)
            store.step = step
            store.auto_state = tracker

            sem_dim = profile.semantic_dim
            emo_dim = profile.emotional_dim
            state_dim = profile.state_dim

            for _ in range(memory_count):
                memory = NCMFile._read_memory(
                    mem_buf,
                    sem_dim,
                    emo_dim,
                    state_dim,
                    fp16=fp16,
                    has_autostate=has_autostate,
                )
                store.add(memory, update_auto_state=False)

            return store
        except CorruptFileError:
            raise
        except Exception as e:
            raise PersistenceError('load', str(e))

    @staticmethod
    def _write_memory(buf, memory, sem_dim, emo_dim, state_dim, fp16: bool = False, has_autostate: bool = False):
        """OPTIMIZATION: Efficient memory serialization with pre-packed vectors."""
        id_bytes = memory.id.encode('utf-8')
        buf.write(struct.pack('>H', len(id_bytes)))
        buf.write(id_bytes)
        buf.write(struct.pack('>q', memory.timestamp))
        buf.write(struct.pack('>f', memory.strength))

        def write_vec(vec, dim):
            """OPTIMIZATION: Avoid intermediate array copy if dimensions match."""
            v = np.asarray(vec, dtype=np.float32)
            if v.shape[0] != dim:
                # Only create temporary if needed
                tmp = np.zeros(dim, dtype=np.float32)
                tmp[:min(len(v), dim)] = v[:min(len(v), dim)]
                v = tmp
            if fp16:
                buf.write(v.astype(np.float16).tobytes())
            else:
                buf.write(v.tobytes())

        write_vec(memory.e_semantic, sem_dim)
        write_vec(memory.e_emotional, emo_dim)
        write_vec(memory.s_snapshot, state_dim)
        if has_autostate:
            auto_state = memory.auto_state_snapshot
            if auto_state is None:
                auto_state = np.full(5, 0.5, dtype=np.float32)
            write_vec(auto_state, 5)

        text_bytes = memory.text.encode('utf-8')[:65535]
        buf.write(struct.pack('>H', len(text_bytes)))
        buf.write(text_bytes)

        # OPTIMIZATION: Filter tags once instead of in comprehension
        valid_tags = [t for t in memory.tags if isinstance(t, str)][:255]
        buf.write(struct.pack('>B', len(valid_tags)))
        for tag in valid_tags:
            tag_bytes = tag.encode('utf-8')[:255]
            buf.write(struct.pack('>B', len(tag_bytes)))
            buf.write(tag_bytes)

    @staticmethod
    def _read_memory(buf, sem_dim, emo_dim, state_dim, fp16: bool = False, has_autostate: bool = False):
        """OPTIMIZATION: Efficient memory deserialization with direct numpy frombuffer."""
        id_len = struct.unpack('>H', buf.read(2))[0]
        memory_id = buf.read(id_len).decode('utf-8')
        timestamp = struct.unpack('>q', buf.read(8))[0]
        strength = struct.unpack('>f', buf.read(4))[0]

        if fp16:
            # 2 bytes per float on disk; cast back to FP32 for in-memory math.
            e_semantic = np.frombuffer(
                NCMFile._read_exact(buf, sem_dim * 2, "e_semantic(fp16)"),
                dtype=np.float16,
            ).astype(np.float32).copy()
            e_emotional = np.frombuffer(
                NCMFile._read_exact(buf, emo_dim * 2, "e_emotional(fp16)"),
                dtype=np.float16,
            ).astype(np.float32).copy()
            s_snapshot = np.frombuffer(
                NCMFile._read_exact(buf, state_dim * 2, "s_snapshot(fp16)"),
                dtype=np.float16,
            ).astype(np.float32).copy()
            auto_state_snapshot = None
            if has_autostate:
                auto_state_snapshot = np.frombuffer(
                    NCMFile._read_exact(buf, 5 * 2, "auto_state_snapshot(fp16)"),
                    dtype=np.float16,
                ).astype(np.float32).copy()
        else:
            # Legacy path: 4 bytes per float on disk.
            e_semantic = np.frombuffer(
                NCMFile._read_exact(buf, sem_dim * 4, "e_semantic(fp32)"),
                dtype=np.float32,
            ).copy()
            e_emotional = np.frombuffer(
                NCMFile._read_exact(buf, emo_dim * 4, "e_emotional(fp32)"),
                dtype=np.float32,
            ).copy()
            s_snapshot = np.frombuffer(
                NCMFile._read_exact(buf, state_dim * 4, "s_snapshot(fp32)"),
                dtype=np.float32,
            ).copy()
            auto_state_snapshot = None
            if has_autostate:
                auto_state_snapshot = np.frombuffer(
                    NCMFile._read_exact(buf, 5 * 4, "auto_state_snapshot(fp32)"),
                    dtype=np.float32,
                ).copy()

        text_len = struct.unpack('>H', buf.read(2))[0]
        text = buf.read(text_len).decode('utf-8')

        tag_count = struct.unpack('>B', buf.read(1))[0]
        tags = []
        for _ in range(tag_count):
            tag_len = struct.unpack('>B', buf.read(1))[0]
            tags.append(buf.read(tag_len).decode('utf-8'))

        return MemoryEntry(
            id=memory_id, e_semantic=e_semantic, e_emotional=e_emotional,
            s_snapshot=s_snapshot, timestamp=timestamp, strength=strength,
            text=text, tags=tags, auto_state_snapshot=auto_state_snapshot,
        )
