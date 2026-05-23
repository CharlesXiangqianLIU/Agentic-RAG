"""Pluggable domain knowledge for the knowledge-rag system.

A *domain pack* is a YAML file that injects domain-specific abbreviations,
synonym groups, fields, unit-normalization regex patterns, and prompt
overrides into ingestion, retrieval, and answer generation. The default
pack is empty — without a pack, the system behaves as a generic RAG.
"""
from domain.loader import DomainPack, load_domain_pack
from domain.cache import get_domain_pack, reset_cache

__all__ = ["DomainPack", "load_domain_pack", "get_domain_pack", "reset_cache"]
