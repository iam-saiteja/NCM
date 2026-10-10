"""NCM - Native Cognitive Memory.

Long-term memory for conversational AI: remembers what was said, recalls what is
relevant, knows who the person is, and forgets what stopped mattering.

    from ncm import Memory
    mem = Memory("alice.ncm")
    mem.add("I adopted a beagle called Max", speaker="user")
    print(mem.context("what's my dog called?"))
    mem.save()
"""

__version__ = "2.0.0"

from ncm.embed import Embedder, FunctionEmbedder
from ncm.memory import Config, Memory, Recalled

__all__ = ["Memory", "Config", "Recalled", "Embedder", "FunctionEmbedder"]
