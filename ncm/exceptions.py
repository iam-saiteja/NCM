"""NCM - Custom exception classes."""


class NCMError(Exception):
    """Base class for all NCM exceptions."""
    pass


class DimensionMismatchError(NCMError):
    def __init__(self, expected, got, context=""):
        self.expected = expected
        self.got = got
        self.context = context
        super().__init__(f"Dimension mismatch in {context}: expected {expected}, got {got}")


class MemoryNotFoundError(NCMError):
    def __init__(self, memory_id):
        self.memory_id = memory_id
        super().__init__(f"Memory not found: {memory_id}")


class EmptyStoreError(NCMError):
    pass


class EncoderNotInitializedError(NCMError):
    pass


class PersistenceError(NCMError):
    def __init__(self, operation, detail=""):
        self.operation = operation
        super().__init__(f"Persistence error during {operation}: {detail}")


class InvalidStateVectorError(NCMError):
    def __init__(self, detail=""):
        super().__init__(f"Invalid state vector: {detail}")


class CorruptFileError(NCMError):
    def __init__(self, path, detail=""):
        self.path = path
        super().__init__(f"Corrupt or unsupported .ncm file at {path}: {detail}")


class ProfileError(NCMError):
    pass
