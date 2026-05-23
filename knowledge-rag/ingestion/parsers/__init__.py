"""Per-format document parsers. Each module defines a ``parse_<fmt>(path)``
function that returns a ``ParsedDocument``. The format-agnostic entry point
is ``ingestion.parser.parse_document``.
"""
