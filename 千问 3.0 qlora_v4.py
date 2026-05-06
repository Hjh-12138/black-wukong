import os, glob, json, datetime

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.tokenization_utils_base import BatchEncoding
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# safetensors mmap patch
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
        import torch
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
        self._key = key; self._info = tensor_info; self._framework = framework; self._device = device
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
    import torch
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

# ── 配置 ──
checkpoint_id = "Qwen/Qwen3-4B-Instruct-2507"
artifacts_dir = "./checkpoints"
data_dir = "./data"
max_seq_len = 2048

os.makedirs(artifacts_dir, exist_ok=True)

# ── 合并所有数据源 ──
print("=== 合并数据集 ===")

all_train_files = sorted(glob.glob(os.path.join(data_dir, "wukong_train_*.jsonl")), key=os.path.getmtime, reverse=True)
all_eval_files  = sorted(glob.glob(os.path.join(data_dir, "wukong_eval_*.jsonl")),  key=os.path.getmtime, reverse=True)
all_knowledge_files = sorted(glob.glob(os.path.join(data_dir, "wukong_knowledge_*.jsonl")), key=os.path.getmtime, reverse=True)

# 使用最新的 train/eval 对（通常是 v4 生成的，因为有最新的时间戳）
train_file = all_train_files[0]
eval_file = all_eval_files[0]
print(f"主训练集: {train_file}")
print(f"验证集: {eval_file}")

# 加载主数据集
train_set = load_dataset("json", data_files=train_file, split="train")
eval_set = load_dataset("json", data_files=eval_file, split="train")

# 合并知识数据集（如果有）
knowledge_set = None
if all_knowledge_files:
    kf = all_knowledge_files[0]
    print(f"知识数据集: {kf}")
    knowledge_set = load_dataset("json", data_files=kf, split="train")
    # 合并
    train_set = concatenate_datasets([train_set, knowledge_set])
    print(f"合并后训练集: {len(train_set)} 条")

# 去重（基于 instruction）
print("去重中...")
seen_q = set()
unique_indices = []
for i, item in enumerate(train_set):
    q = (item["instruction"] or "").strip()
    if q and q not in seen_q:
        seen_q.add(q)
        unique_indices.append(i)

train_set = train_set.select(unique_indices)
print(f"去重后训练集: {len(train_set)} 条")
print(f"验证集: {len(eval_set)} 条")

# ── 分词器 & 模型 ──
tokenizer = AutoTokenizer.from_pretrained(checkpoint_id, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"pad={tokenizer.pad_token_id} eos={tokenizer.eos_token_id}")

compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=compute_dtype,
)

base_model = AutoModelForCausalLM.from_pretrained(
    checkpoint_id,
    trust_remote_code=True,
    quantization_config=bnb_cfg,
    device_map="cuda:0",
)
base_model.config.use_cache = False
base_model.gradient_checkpointing_enable()
base_model = prepare_model_for_kbit_training(base_model)

# ── LoRA: 覆盖所有线性层，扩大学习容量 ──
lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
peft_model = get_peft_model(base_model, lora_cfg)
peft_model.enable_input_require_grads()
peft_model.config.use_cache = False
peft_model.print_trainable_parameters()

# ── 数据格式化 ──
from typing import List, Dict
from torch.utils.data import Dataset as TorchDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["DATASETS_PROCESSING_MULTIPROCESSING"] = "0"

SYSTEM_MSG = "你是《黑神话：悟空》领域助手，回答准确、简明。"

def format_sample(record):
    try:
        instr = (record.get("instruction") or "").strip()
        ans = (record.get("output") or "").strip()
        if not instr or not ans:
            return {"input_ids": [], "labels": []}

        msgs_no_assist = [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": instr},
        ]
        prompt_ids = tokenizer.apply_chat_template(
            msgs_no_assist, tokenize=True, add_generation_prompt=True, return_tensors=None,
        )
        if isinstance(prompt_ids, BatchEncoding):
            prompt_ids = prompt_ids['input_ids']
        elif hasattr(prompt_ids, 'ids'):
            prompt_ids = prompt_ids.ids
        prompt_ids = prompt_ids[:max_seq_len]

        msgs_full = msgs_no_assist + [{"role": "assistant", "content": ans}]
        full_ids = tokenizer.apply_chat_template(
            msgs_full, tokenize=True, add_generation_prompt=False, return_tensors=None,
        )
        if isinstance(full_ids, BatchEncoding):
            full_ids = full_ids['input_ids']
        elif hasattr(full_ids, 'ids'):
            full_ids = full_ids.ids
        full_ids = full_ids[:max_seq_len]

        cut = len(prompt_ids)
        if cut >= len(full_ids):
            labels = [-100] * len(full_ids)
        else:
            labels = [-100] * cut + full_ids[cut:]

        return {"input_ids": list(full_ids), "labels": list(labels)}
    except Exception as e:
        print(f"Skip: {e}")
        return {"input_ids": [], "labels": []}


class QwenSftDataset(TorchDataset):
    def __init__(self, hf_dataset, format_fn):
        self.data = []
        total = len(hf_dataset)
        for i, record in enumerate(hf_dataset):
            if (i + 1) % 300 == 0:
                print(f"  [{i+1}/{total}]")
            result = format_fn(record)
            if len(result.get("input_ids", [])) > 0:
                self.data.append(result)
        print(f"Valid: {len(self.data)}/{total}")
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]


class QwenSftCollator:
    def __init__(self, pad_id: int, max_length: int = 2048, ignore_id: int = -100):
        self.pad_id = pad_id; self.max_length = max_length; self.ignore_id = ignore_id
    def __call__(self, features: List[Dict]):
        max_len = max(len(f["input_ids"]) for f in features)
        max_len = min(max_len, self.max_length)
        input_ids, labels = [], []
        for f in features:
            ids = f["input_ids"][:max_len]
            lbs = f["labels"][:max_len]
            pad = max_len - len(ids)
            if pad > 0:
                ids = ids + [self.pad_id] * pad
                lbs = lbs + [self.ignore_id] * pad
            input_ids.append(torch.tensor(ids, dtype=torch.long))
            labels.append(torch.tensor(lbs, dtype=torch.long))
        return {"input_ids": torch.stack(input_ids), "labels": torch.stack(labels)}


print("\nProcessing train dataset...")
proc_train = QwenSftDataset(train_set, format_sample)
print("Processing eval dataset...")
proc_eval  = QwenSftDataset(eval_set, format_sample)

collator = QwenSftCollator(pad_id=tokenizer.pad_token_id, max_length=max_seq_len)

# ── 训练参数 ──
now_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = os.path.join(artifacts_dir, f"qwen25_wukong_lora_{now_tag}")

from transformers import TrainingArguments, Trainer

args = TrainingArguments(
    output_dir=run_dir,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=3e-4,              # 5e-4 → 3e-4 更稳定
    num_train_epochs=4,              # 3 → 4（更多数据时需要更多 epoch）
    lr_scheduler_type="linear",
    warmup_ratio=0.05,
    weight_decay=0.01,
    logging_steps=1,
    save_steps=50,
    save_total_limit=3,
    eval_strategy="steps",
    eval_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    optim="adamw_torch",
    bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    fp16=not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
    report_to=[],
    ddp_find_unused_parameters=False,
)

trainer = Trainer(
    model=peft_model,
    args=args,
    train_dataset=proc_train,
    eval_dataset=proc_eval,
    data_collator=collator,
)

# ── 开始训练 ──
train_output = trainer.train()
print(train_output)

trainer.save_model(run_dir)
tokenizer.save_pretrained(run_dir)
print(f"Saved to: {run_dir}")

stats = {
    "run_dir": run_dir,
    "train_samples": len(proc_train),
    "eval_samples": len(proc_eval),
    "epochs": 4,
    "learning_rate": 3e-4,
    "weight_decay": 0.01,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "final_train_loss": train_output.training_loss if hasattr(train_output, "training_loss") else None,
    "best_eval_loss": trainer.state.best_metric if hasattr(trainer.state, "best_metric") else None,
}
with open(os.path.join(run_dir, "training_stats.json"), "w") as f:
    json.dump(stats, f, indent=2)
print(f"Stats: {stats}")
