import jieba


def retrieve(query: str, bm25, corpus_meta: list, threshold: float, top_k: int = 4):
    """检索并返回 top_k 结果，同时过滤低于阈值的。返回 list[dict]"""
    tokens = jieba.lcut(query)
    scores = bm25.get_scores(tokens)

    if len(scores) == 0:
        return [], False

    indexed = [(i, scores[i]) for i in range(len(scores))]
    indexed.sort(key=lambda x: x[1], reverse=True)

    top_score = indexed[0][1]
    below_threshold = top_score < threshold

    results = []
    for idx, score in indexed[:top_k]:
        if score <= 0:
            break
        results.append({
            "idx": idx,
            "score": round(float(score), 2),
            "source": corpus_meta[idx]["source"],
            "type": corpus_meta[idx]["type"],
        })

    return results, below_threshold
