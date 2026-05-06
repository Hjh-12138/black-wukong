import os, re, json, datetime, time, random, glob
from openai import OpenAI

"""
改进版 v4：多 Q&A 生成 + 减少 paraphrase，大幅提升唯一答案数
===========================================================
核心改进：
  1. 每段生成 3 个 Q&A pairs（而非 1 个）→ 唯一答案数 ×3
  2. 减少 paraphrase 到 1 个变体（而非 3 个）
  3. 降低学习率（训练脚本层面）
"""

DATA_DIR = "./data"
TS = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

OUT_BASE_JSONL = f"{DATA_DIR}/wukong_base_{TS}.jsonl"
OUT_TRAIN_JSONL = f"{DATA_DIR}/wukong_train_{TS}.jsonl"
OUT_EVAL_JSONL = f"{DATA_DIR}/wukong_eval_{TS}.jsonl"

# ── 配置 ──
NUM_QA_PER_SEGMENT = 3   # 每段生成 3 个 Q&A
NUM_VARIANTS = 1          # 每个问题只生成 1 个 paraphrase
EVAL_RATIO = 0.10
MIN_SEG_LEN = 30
MAX_SEG_LEN = 5000

BASE_URL = "https://api.deepseek.com"
MODEL_ID = "deepseek-v4-flash"
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-xxx")
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def smart_split(content: str, min_len=MIN_SEG_LEN, max_len=MAX_SEG_LEN):
    """智能切分：支持 h2/h3/h4 + 粗体关键词拆分"""
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


# ── 读取并切分所有 MD 文件 ──
md_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.md")))
if not md_files:
    raise FileNotFoundError(f"No .md files in {DATA_DIR}")
print(f"MD 文件: {len(md_files)} 个")

all_sections = []
for md_path in md_files:
    with open(md_path, "r", encoding="utf-8") as f:
        raw_md = f.read()
    sections = smart_split(raw_md)
    all_sections.extend(sections)
    print(f"  {os.path.basename(md_path)}: {len(sections)} 段")

# 去重
seen = set()
uniq = []
for t in all_sections:
    key = re.sub(r"\s+", " ", t).lower()[:240]
    if key not in seen:
        seen.add(key)
        uniq.append(t)
all_sections = uniq
print(f"去重后共 {len(all_sections)} 段")


# ─────────────────────────────────────────────
#  教师模型：每段生成多个 Q&A pairs
# ─────────────────────────────────────────────
MULTI_QA_SYSTEM_PROMPT = (
    "你是《黑神话：悟空》的资深资料整理者。"
    f"根据给定原文片段，生成 {NUM_QA_PER_SEGMENT} 个不同的问答对。"
    "严格输出 JSON："
    '{"qa_pairs": [{"instruction":"问题1","output":"答案1"}, ...]}。'
    "要求："
    "1. 每个问题必须是不同角度（如：是什么、怎么获得、在哪里、有什么效果、如何解锁）；"
    "2. 每个答案只依据原文，不要臆测；"
    "3. 答案要完整准确，包含关键细节；"
    "4. 禁止任何额外说明或代码块。"
)

def call_api(messages, temperature=0.2, max_tokens=1200):
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL_ID,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                extra_body={"thinking": {"type": "disabled"}},
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"  API attempt {attempt+1}: {e}")
            if attempt == 2:
                return None
            time.sleep(1.5 ** attempt + random.random() * 0.3)

os.makedirs(DATA_DIR, exist_ok=True)
base_written = 0

with open(OUT_BASE_JSONL, "w", encoding="utf-8") as fbase:
    for i, seg in enumerate(all_sections):
        print(f"  [{i+1}/{len(all_sections)}] 生成多 Q&A...", end="", flush=True)
        content = call_api([
            {"role": "system", "content": MULTI_QA_SYSTEM_PROMPT},
            {"role": "user", "content": seg},
        ])
        if not content:
            print(" 跳过（API 失败）")
            continue
        try:
            obj = json.loads(content)
            qa_list = obj.get("qa_pairs", [])
            if not qa_list:
                print(" 空结果")
                continue

            count = 0
            for qa in qa_list:
                ins = (qa.get("instruction") or "").strip()
                out = (qa.get("output") or "").strip()
                if len(ins) < 8 or len(out) < 10:
                    continue
                if not ins.endswith(("?", "？")):
                    ins += "？"
                fbase.write(json.dumps({"instruction": ins, "output": out}, ensure_ascii=False) + "\n")
                count += 1

            base_written += count
            print(f" → +{count} Q&A")
        except (json.JSONDecodeError, AttributeError, KeyError, TypeError) as e:
            print(f" 解析失败: {e}")
            continue
        time.sleep(0.3)

print(f"\n生成 base Q&A: {base_written} 条 → {OUT_BASE_JSONL}")


# ─────────────────────────────────────────────
#  验证集拆分
# ─────────────────────────────────────────────
with open(OUT_BASE_JSONL, "r", encoding="utf-8") as f:
    base_pairs = [json.loads(line) for line in f if line.strip()]

random.seed(42)
random.shuffle(base_pairs)
split_idx = max(1, int(len(base_pairs) * EVAL_RATIO))
eval_pairs = base_pairs[:split_idx]
train_pairs = base_pairs[split_idx:]
print(f"切分: {len(train_pairs)} 训练 / {len(eval_pairs)} 验证")


# ─────────────────────────────────────────────
#  Paraphrase（仅训练集，少量）
# ─────────────────────────────────────────────
PARAPHRASE_PROMPT_TMPL = (
    '你是一位擅长出题的老师。根据给定「基础问题」，生成{n}个语义等价但表达方式'
    '不同的问句。要求：\n'
    '1. 每个问句必须是完整、可直接回答的自然语言问题\n'
    '2. 覆盖不同的提问角度：是什么、怎么获得、在哪里、有什么作用\n'
    '3. 严格输出 JSON: {{"paraphrases": ["...", ...]}}\n'
    '4. 禁止任何额外文本\n'
)

def generate_paraphrases(base_q: str, n: int) -> list:
    content = call_api([
        {"role": "system", "content": PARAPHRASE_PROMPT_TMPL.format(n=n)},
        {"role": "user", "content": f"基础问题：{base_q}"},
    ], temperature=0.7, max_tokens=800)
    if not content:
        return []
    try:
        obj = json.loads(content)
        arr = obj.get("paraphrases", [])
        arr = [x.strip() for x in arr if isinstance(x, str) and x.strip()]
        arr = [x for x in arr if x.endswith(("?", "？"))]
        return arr[:n]
    except (json.JSONDecodeError, AttributeError, KeyError):
        return []


written_train = 0
seen_q = set()

with open(OUT_TRAIN_JSONL, "w", encoding="utf-8") as ftrain, \
     open(OUT_EVAL_JSONL, "w", encoding="utf-8") as feval:

    for pair in eval_pairs:
        feval.write(json.dumps(pair, ensure_ascii=False) + "\n")

    for pair in train_pairs:
        base_q = pair["instruction"]
        answer = pair["output"]

        if base_q not in seen_q:
            seen_q.add(base_q)
            ftrain.write(json.dumps(pair, ensure_ascii=False) + "\n")
            written_train += 1

        # 少量 paraphrase
        print(f"  paraphrase: {base_q[:40]}...", end="", flush=True)
        variants = generate_paraphrases(base_q, NUM_VARIANTS)
        for v in variants:
            if v not in seen_q:
                seen_q.add(v)
                ftrain.write(json.dumps({"instruction": v, "output": answer}, ensure_ascii=False) + "\n")
                written_train += 1
        print(f" → +{len(variants)} 变体")
        time.sleep(0.3)

print(f"\n训练集: {written_train} 条 → {OUT_TRAIN_JSONL}")
print(f"验证集: {len(eval_pairs)} 条 → {OUT_EVAL_JSONL}")

# 保存元信息
meta = {
    "timestamp": TS,
    "md_files": [os.path.basename(f) for f in md_files],
    "total_sections": len(all_sections),
    "total_base_pairs": base_written,
    "train_base_count": len(train_pairs),
    "eval_base_count": len(eval_pairs),
    "train_samples": written_train,
    "eval_samples": len(eval_pairs),
    "num_qa_per_segment": NUM_QA_PER_SEGMENT,
    "num_variants": NUM_VARIANTS,
    "min_seg_len": MIN_SEG_LEN,
    "max_seg_len": MAX_SEG_LEN,
}
with open(f"{DATA_DIR}/wukong_meta_{TS}.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print(f"元信息已保存")
print("完成！")
