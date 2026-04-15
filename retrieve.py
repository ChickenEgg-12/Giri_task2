"""
Implement your retrieval system here.

The test_files/ folder contains mixed-format documents:
  - PDFs, PowerPoint, Word, emails, JSON glossary

Your retrieve() function should handle both:
  - Broad queries: "What was the research methodology?"
  - Narrow queries: "What does 'frm_brand_awareness' measure?"
"""


def retrieve(query: str) -> list[dict]:
    """
    Return top-5 most relevant passages for the query.

    Each result must have:
      - "text": str       — the passage content
      - "source": str     — source filename or "glossary"
      - "score": float    — relevance score (higher = better)
    """
    # TODO: implement your retrieval system
    return []
