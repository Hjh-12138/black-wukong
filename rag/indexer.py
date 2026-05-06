import os, re, json, glob
import jieba
from rank_bm25 import BM25Okapi

MIN_SEG_LEN = 30
MAX_SEG_LEN = 5000


def smart_split(content: str, min_len=MIN_SEG_LEN, max_len=MAX_SEG_LEN):
    sections = []
    headings = list(re.finditer(r"(?m)^(#{2,4})\s+(.+)$", content))

    if headings:
        for i, m in enumerate(headings):
            start = m.start()
            end = headings[i + 1].start() if i + 1 < len(headings) else len(content)
            block = content[start:end].strip()
            if len(block) >= min_len:
                sub_blocks = re.split(r"\n(?=\*\*[^*]+\*\*[:：])", block)
                if len(sub_blocks) > 1:
                    for sb in sub_blocks:
                        sb = sb.strip()
                        if min_len <= len(sb) <= max_len:
                            sections.append(sb)
                else:
                    sections.append(block)
    else:
        paras = [p.strip() for p in re.split(r"\n\s*\n", content)
                 if min_len <= len(p.strip()) <= max_len]
        sections.extend(paras)

    final = []
    for s in sections:
        if len(s) > max_len:
            paras = [p.strip() for p in re.split(r"\n\s*\n", s)
                     if len(p.strip()) >= min_len]
            final.extend(paras)
        else:
            final.append(s)
    return final


def build_index(data_dir="./data"):
    """构建 BM25 索引，返回 (bm25, corpus_meta, threshold)"""

    docs = []
    meta = []

    # ── 1. 加载 MD 文件并切分 ──
    md_files = sorted(glob.glob(os.path.join(data_dir, "*.md")))
    if md_files:
        for md_path in md_files:
            with open(md_path, "r", encoding="utf-8") as f:
                raw = f.read()
            sections = smart_split(raw)
            fname = os.path.basename(md_path)
            for sec in sections:
                docs.append(sec)
                meta.append({"source": fname, "type": "md"})
        print(f"[indexer] MD 文件 {len(md_files)} 个 → {len(docs)} 段")

    # ── 2. 加载 JSONL Q&A 数据 ──
    jsonl_files = sorted(
        glob.glob(os.path.join(data_dir, "wukong_*.jsonl")),
        key=os.path.getmtime, reverse=True
    )
    qa_count = 0
    seen_q = set()
    for jf in jsonl_files:
        fname = os.path.basename(jf)
        with open(jf, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
                instr = (rec.get("instruction") or "").strip()
                output = (rec.get("output") or "").strip()
                if not instr or not output:
                    continue
                if instr in seen_q:
                    continue
                seen_q.add(instr)
                text = f"问：{instr}\n答：{output}"
                docs.append(text)
                meta.append({"source": fname, "type": "qa"})
                qa_count += 1

    print(f"[indexer] JSONL 文件 → {qa_count} 条 Q&A (去重后)")

    # ── 3. MD / Q&A 跨界去重 ──
    # 仅针对 MD 文档，检查是否与 Q&A 的 output 高度重叠
    # 简化：按前 120 字符去空格后比较
    md_sigs = set()
    keep_indices = []
    for i, doc in enumerate(docs):
        if meta[i]["type"] != "md":
            keep_indices.append(i)
            continue
        sig = re.sub(r"\s+", "", doc)[:120]
        if sig not in md_sigs:
            md_sigs.add(sig)
            keep_indices.append(i)
        # else skip duplicate MD segment

    docs = [docs[i] for i in keep_indices]
    meta = [meta[i] for i in keep_indices]
    print(f"[indexer] 去重后共 {len(docs)} 条知识")

    # ── 4. 分词 + 构建 BM25 ──
    tokenized = [jieba.lcut(doc) for doc in docs]
    bm25 = BM25Okapi(tokenized)

    # ── 5. 自动阈值：用几个校准问题取中位数 ──
    calibration_questions = [
        "如何获得出云棍？",
        "黑风大王在哪里？",
        "定身术怎么升级？",
        "盘丝岭有哪些宝物？",
        "混铁棍有什么效果？",
    ]
    all_scores = []
    for q in calibration_questions:
        q_tokens = jieba.lcut(q)
        scores = bm25.get_scores(q_tokens)
        top1 = max(scores) if len(scores) > 0 else 0
        if top1 > 0:
            all_scores.append(top1)

    if all_scores:
        all_scores.sort()
        threshold = max(all_scores[0] * 0.3, 0.5)
    else:
        threshold = 0.5

    print(f"[indexer] BM25 索引就绪，阈值 = {threshold:.2f} "
          f"(最低校准分 {all_scores[0]:.1f} x 0.3)")
    return bm25, meta, docs, threshold
