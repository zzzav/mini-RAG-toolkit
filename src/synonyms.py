DEFAULT_SYNONYMS = {
    "invoice": ["bill"],
    "bill": ["invoice"],
    "refund": ["reimbursement"],
    "reimbursement": ["refund"],
    "salary": ["payroll"],
    "payroll": ["salary"],
}


def expand_tokens(
    tokens: list[str], synonyms: dict[str, list[str]] | None = None, max_expansions: int = 3
) -> list[str]:
    out_tokens: list[str] = []

    for t in tokens:

        if t not in out_tokens:
            out_tokens.append(t)

        if not synonyms:
            continue

        iSynonyms = synonyms.get(t, [])
        if iSynonyms == []:
            continue

        for s in iSynonyms[:max_expansions]:
            if s not in out_tokens:
                out_tokens.append(s)

    return out_tokens
