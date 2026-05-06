def build_rag_prompt(user_query: str, retrieved_docs: list[dict], corpus_texts: list[str]):
    """构建注入检索上下文的用户消息，返回 (full_user_message, source_lines)"""

    if not retrieved_docs:
        return user_query, []

    ref_parts = []
    source_lines = []
    for i, doc in enumerate(retrieved_docs):
        idx = doc["idx"]
        source = doc["source"]
        text = corpus_texts[idx]
        snippet = text[:80].replace("\n", " ")
        ref_parts.append(f"[来源{i + 1}: {source}]\n{text}")
        source_lines.append(f"{i + 1}. [{source}] {snippet}...")

    ref_block = "\n\n---\n\n".join(ref_parts)

    full_message = (
        f"【参考资料】\n"
        f"---\n"
        f"{ref_block}\n"
        f"---\n\n"
        f"【问题】\n"
        f"{user_query}\n\n"
        f"请根据以上参考资料回答问题。如果资料不足以回答，请明确说明。"
    )

    return full_message, source_lines
