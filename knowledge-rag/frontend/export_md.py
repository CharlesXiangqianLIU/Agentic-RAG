# frontend/export_md.py
"""Export full conversation history as Markdown."""
from datetime import datetime


def conversation_to_markdown(history: list[dict]) -> bytes:
    """Convert session history to a Markdown document.

    Each assistant message becomes a Q&A block:
    ## Q: {question}
    {answer}
    ### Sources
    - [Source: file | Page N | Section: S]

    Args:
        history: list of dicts with keys: role, content, question, chunks, question_type

    Returns:
        UTF-8 encoded bytes of the Markdown document.
    """
    lines = [
        "# Knowledge Base — Conversation Export",
        f"_Exported: {datetime.now().strftime('%Y-%m-%d %H:%M')}_",
        "",
    ]

    for msg in history:
        if msg["role"] == "assistant" and msg.get("question"):
            lines.append(f"## Q: {msg['question']}")
            lines.append("")
            lines.append(msg.get("content", ""))
            lines.append("")

            if msg.get("question_type"):
                lines.append(f"_Query type: {msg['question_type']}_")
                lines.append("")

            chunks = msg.get("chunks", [])
            if chunks:
                lines.append("### Sources")
                seen = set()
                for chunk in chunks:
                    attr = chunk.get("attribution", "")
                    if attr and attr not in seen:
                        seen.add(attr)
                        lines.append(f"- {attr}")
                lines.append("")

            lines.append("---")
            lines.append("")

    return "\n".join(lines).encode("utf-8")
