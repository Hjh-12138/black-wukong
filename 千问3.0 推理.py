import os, sys, argparse

# Windows 控制台 UTF-8 编码
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_CACHE", "D:/models/hub/hub")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ── safetensors mmap patch ──
import transformers.modeling_utils as _mu
import safetensors


class _NoMmapSafeOpen:
    def __init__(self, file, framework="pt", device="cpu"):
        with open(file, "rb") as f:
            raw = f.read()
        self._tensors = safetensors.deserialize(raw)
        self._framework = framework
        self._device = device

    def __enter__(self): return self

    def __exit__(self, *args): self._tensors = None

    def keys(self): return [name for name, _ in self._tensors]

    def get_slice(self, key):
        for name, tensor_info in self._tensors:
            if name == key:
                return _TensorSlice(key, tensor_info, self._framework, self._device)
        raise KeyError(f"Tensor {key} not found")

    def get_tensor(self, key):
        for name, tensor_info in self._tensors:
            if name == key:
                dtype = tensor_info["dtype"]
                shape = tensor_info["shape"]
                data = tensor_info["data"]
                torch_dtype = _safetensor_dtype_to_torch(dtype)
                t = torch.frombuffer(bytearray(data), dtype=torch_dtype).reshape(shape)
                if self._device and self._device != "cpu":
                    t = t.to(self._device)
                return t
        raise KeyError(f"Tensor {key} not found")


class _TensorSlice:
    def __init__(self, key, tensor_info, framework, device):
        self._key = key
        self._info = tensor_info
        self._framework = framework
        self._device = device
        self._tensor = None

    def _load(self):
        if self._tensor is None:
            dtype = _safetensor_dtype_to_torch(self._info["dtype"])
            shape = list(self._info["shape"])
            self._tensor = torch.frombuffer(bytearray(self._info["data"]), dtype=dtype).reshape(shape)
            if self._device and self._device != "cpu":
                self._tensor = self._tensor.to(self._device)
        return self._tensor

    def __getitem__(self, idx): return self._load()[idx]

    def get_dtype(self): return self._info["dtype"]

    def get_shape(self): return list(self._info["shape"])


def _safetensor_dtype_to_torch(dtype_str):
    mapping = {"F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
               "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32,
               "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool}
    return mapping.get(dtype_str, torch.float32)


_mu.safe_open = _NoMmapSafeOpen
_original_load_state_dict = _mu.load_state_dict


def _patched_load_state_dict(checkpoint_file, map_location="cpu", weights_only=True):
    if str(checkpoint_file).endswith(".safetensors"):
        with _NoMmapSafeOpen(checkpoint_file, framework="pt", device="cpu") as f:
            state_dict = {}
            for k in f.keys():
                if map_location == "meta":
                    _slice = f.get_slice(k)
                    k_dtype = _slice.get_dtype()
                    dtype = _safetensor_dtype_to_torch(k_dtype)
                    state_dict[k] = torch.empty(size=_slice.get_shape(), dtype=dtype, device="meta")
                else:
                    state_dict[k] = f.get_tensor(k).to(map_location)
            return state_dict
    return _original_load_state_dict(checkpoint_file, map_location, weights_only)


_mu.load_state_dict = _patched_load_state_dict

# ── RAG 模块 ──
from rag import build_index, retrieve, build_rag_prompt

# ── 配置 ──
base_model_id = "Qwen/Qwen3-4B-Instruct-2507"
lora_adapter_path = "./checkpoints/qwen25_wukong_lora_20260506_155303/checkpoint-150"
DATA_DIR = "./data"

# ── CLI ──
parser = argparse.ArgumentParser(description="黑神话悟空 Qwen3 LoRA 推理 (RAG)")
parser.add_argument("question", nargs="?", default=None, help="单次推理问题")
parser.add_argument("--top-k", type=int, default=4, help="检索片段数量 (默认 4)")
parser.add_argument("--rebuild-index", action="store_true", help="强制重建索引")
args = parser.parse_args()

# ── 构建 RAG 索引 ──
print("=== 构建知识库索引 ===")
bm25, corpus_meta, corpus_texts, threshold = build_index(DATA_DIR)


# ── 加载分词器 ──
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    trust_remote_code=True,
)

# ── 加载基座模型（4-bit） ──
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map="cuda:0" if torch.cuda.is_available() else "cpu",
)

# ── 加载 LoRA 适配器 ──
model = PeftModel.from_pretrained(base_model, lora_adapter_path)
model.eval()
print(f"LoRA 适配器已加载。可训练参数: {model.peft_config['default'].r}")

# ── 推理 ──
NOT_FOUND_MSG = "抱歉，当前知识库中没有找到关于这个问题的相关信息。"


def chat(user_message, system_message="你是《黑神话：悟空》领域助手，回答准确、简明。",
         use_rag=True, top_k=4):
    sources = []

    if use_rag:
        # 检索
        results, below_threshold = retrieve(user_message, bm25, corpus_meta, threshold, top_k)

        if below_threshold or not results:
            return NOT_FOUND_MSG, []

        # 获取原文并构建 RAG prompt
        rag_message, sources = build_rag_prompt(user_message, results, corpus_texts)
        user_message = rag_message

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05,
            do_sample=True,
        )

    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response, sources


def format_output(response: str, sources: list) -> str:
    """格式化输出，附加来源信息"""
    out = response
    if sources:
        out += "\n\n---\n[参考来源]:\n"
        out += "\n".join(sources)
    return out


if __name__ == "__main__":
    if args.question:
        answer, sources = chat(args.question, top_k=args.top_k)
        print(format_output(answer, sources))
        sys.exit(0)

    print("\n" + "=" * 50)
    print("黑神话悟空 Qwen3 LoRA 推理 (RAG)")
    print(f"模型: {lora_adapter_path}")
    print(f"检索片段数: {args.top_k}  阈值: {threshold:.2f}")
    print("=" * 50)

    while True:
        try:
            user_input = input("\n问题: ").strip()
            if user_input.lower() in ("exit", "quit", "exit()", "quit()"):
                break
            if not user_input:
                continue

            answer, sources = chat(user_input, top_k=args.top_k)
            print(f"\n{format_output(answer, sources)}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n错误: {e}")
