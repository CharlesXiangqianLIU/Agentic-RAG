"""Process-wide singleton accessor for the configured domain pack.

`get_domain_pack()` reads `config.DOMAIN_PACK_PATH` exactly once per
process and returns the resulting `DomainPack`. Tests may call
`reset_cache()` to force a fresh load (typically after patching the
env or the config module).
"""
from __future__ import annotations

from functools import lru_cache

from domain.loader import DomainPack, load_domain_pack


@lru_cache(maxsize=1)
def get_domain_pack() -> DomainPack:
    import config  # local import to allow tests to monkeypatch config.DOMAIN_PACK_PATH first

    return load_domain_pack(config.DOMAIN_PACK_PATH)


def reset_cache() -> None:
    get_domain_pack.cache_clear()
