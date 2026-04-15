#!/usr/bin/env python
"""Local test harness. Keyword-based scorer (not the real eval). Run unlimited.

Tests both document retrieval (broad) and glossary lookup (narrow).
The real evaluation uses different queries and LLM-as-judge scoring.
"""

QUERIES = [
    # Broad (document retrieval)
    {"id": "b1", "type": "broad", "query": "What was the main purpose of the consumer research study?"},
    {"id": "b2", "type": "broad", "query": "What methodology was used in the research?"},
    {"id": "b3", "type": "broad", "query": "What are the key findings about brand perception?"},
    {"id": "b4", "type": "broad", "query": "What consumer segments were identified?"},
    {"id": "b5", "type": "broad", "query": "What is the competitive landscape for frozen meals?"},
    # Narrow (glossary + specific lookups)
    {"id": "n1", "type": "narrow", "query": "What does 'frm_brand_awareness' measure?"},
    {"id": "n2", "type": "narrow", "query": "What is the total sample size of the study?"},
    {"id": "n3", "type": "narrow", "query": "What does 'probable trialists' mean in this context?"},
    {"id": "n4", "type": "narrow", "query": "What scale was used for preference ratings?"},
    {"id": "n5", "type": "narrow", "query": "Define 'share of wallet' as used in this study"},
]

STOPWORDS = {"what", "is", "the", "a", "an", "of", "in", "for", "was", "were",
             "how", "does", "do", "this", "that", "give", "me", "are", "about",
             "used", "mean", "define", "as"}


def keyword_score(query: str, passages: list[dict]) -> float:
    if not passages:
        return 0.0
    terms = set(query.lower().split()) - STOPWORDS
    if not terms:
        return 0.5
    best = 0.0
    for p in passages[:5]:
        text = p.get("text", "").lower()
        chunk_terms = set(text.split())
        overlap = len(terms & chunk_terms)
        best = max(best, overlap / len(terms))
    return min(best, 1.0)


def main():
    from retrieve import retrieve

    broad_scores, narrow_scores = [], []

    print(f"\n{'='*65}")
    print(f"  Local Test (keyword scorer — not the real evaluation)")
    print(f"{'='*65}\n")

    for q in QUERIES:
        try:
            results = retrieve(q["query"])
        except Exception as e:
            print(f'  [ERR ] {q["type"]:6s} 0.00 | {q["query"][:45]:45s} | {e}')
            (broad_scores if q["type"] == "broad" else narrow_scores).append(0.0)
            continue

        score = keyword_score(q["query"], results)
        bucket = broad_scores if q["type"] == "broad" else narrow_scores
        bucket.append(score)
        mark = "OK" if score > 0.3 else "MISS"
        src = results[0]["source"][:30] if results else "NONE"
        print(f'  [{mark:4s}] {q["type"]:6s} {score:.2f} | {q["query"][:45]:45s} | {src}')

    broad = sum(broad_scores) / len(broad_scores) if broad_scores else 0
    narrow = sum(narrow_scores) / len(narrow_scores) if narrow_scores else 0
    overall = (broad + narrow) / 2

    print(f"\n{'='*65}")
    print(f"  Broad:   {broad:.1%}")
    print(f"  Narrow:  {narrow:.1%}")
    print(f"  Overall: {overall:.1%}")
    print(f"{'='*65}")
    print(f"\n  Note: keyword scorer only. Real eval uses LLM-as-judge")
    print(f"  and different queries. This is for development iteration.\n")


if __name__ == "__main__":
    main()
