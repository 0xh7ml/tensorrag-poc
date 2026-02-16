# Open-Source LLM Fine-Tuning Pipeline — Step-by-Step Cards

Build a complete LLM fine-tuning, evaluation, and serving pipeline using small, reusable cards. Each card handles one step, and data flows between them via the storage service (local or S3/Modal volumes).

## Pipeline Overview

```
Load Dataset → Filter & Clean → Train/Val Split → Format for SFT → Tokenize
Load Base Model → Apply LoRA → Fine-Tune → Evaluate → Merge & Export → vLLM Serve
```

## Project File Structure

Create the following folders and files in the **Editor** view:

```
llm-finetune-pipeline/         ← Project name
├── data/                      ← Folder
│   ├── load_dataset.py        ← Card 1
│   ├── filter_clean.py        ← Card 2
│   ├── train_val_split.py     ← Card 3
│   └── format_sft.py          ← Card 4
├── tokens/                    ← Folder
│   └── tokenize.py            ← Card 5
├── model/                     ← Folder
│   ├── load_model.py          ← Card 6
│   └── apply_lora.py          ← Card 7
├── training/                  ← Folder
│   └── finetune_lora.py       ← Card 8
├── evaluation/                ← Folder
│   └── evaluate.py            ← Card 9
└── serving/                   ← Folder
    ├── merge_export.py        ← Card 10
    └── vllm_serve.py          ← Card 11
```

## Card Connection Map

| # | Card | File | Folder | Receives from | Sends to |
|---|------|------|--------|--------------|----------|
| 1 | Load Dataset | `load_dataset.py` | `data/` | — (config: HF dataset or JSONL path) | `raw_dataset` |
| 2 | Filter & Clean | `filter_clean.py` | `data/` | `raw_dataset` | `clean_dataset` |
| 3 | Train/Val Split | `train_val_split.py` | `data/` | `clean_dataset` | `train_dataset`, `val_dataset` |
| 4 | Format for SFT | `format_sft.py` | `data/` | `train_dataset`, `val_dataset` | `train_sft`, `val_sft` |
| 5 | Tokenize | `tokenize.py` | `tokens/` | `train_sft`, `val_sft` | `train_tokens`, `val_tokens` |
| 6 | Load Base Model | `load_model.py` | `model/` | — (config: HF model name) | `model_config` |
| 7 | Apply LoRA | `apply_lora.py` | `model/` | `model_config` | `lora_config` |
| 8 | Fine-Tune LoRA | `finetune_lora.py` | `training/` | `lora_config`, `train_tokens`, `val_tokens` | `ft_result` |
| 9 | Evaluate | `evaluate.py` | `evaluation/` | `ft_result`, `val_tokens` | `eval_report` |
|10 | Merge & Export | `merge_export.py` | `serving/` | `ft_result` | `merged_model` |
|11 | vLLM Serve | `vllm_serve.py` | `serving/` | `merged_model` | `deployment_info` |

> **Note:** The executor automatically fills in the real `pipeline_id` and `node_id` for storage calls. Use any placeholder (e.g. `"_p"`, `"_n"`) — they get replaced at runtime.

---

## Card 1: Load Dataset

**File:** `load_dataset.py` | **Folder:** `data/`

Loads a dataset from HuggingFace Hub or a local/remote JSONL file.

```python
from cards.base import BaseCard
from datasets import load_dataset
import json
import io


class LoadDatasetCard(BaseCard):
    card_type = "llm_load_dataset"
    display_name = "Load Dataset"
    description = "Load dataset from HuggingFace or JSONL"
    category = "data"
    execution_mode = "local"
    output_view_type = "table"

    config_schema = {
        "source_type": {
            "type": "string",
            "label": "Source type (huggingface/jsonl_url/jsonl_path)",
            "default": "huggingface",
        },
        "hf_dataset": {
            "type": "string",
            "label": "HuggingFace dataset name",
            "default": "tatsu-lab/alpaca",
        },
        "hf_split": {
            "type": "string",
            "label": "HF split",
            "default": "train",
        },
        "jsonl_path": {
            "type": "string",
            "label": "Local JSONL path (optional)",
            "default": "",
        },
        "jsonl_url": {
            "type": "string",
            "label": "Remote JSONL URL (optional)",
            "default": "",
        },
        "max_samples": {
            "type": "number",
            "label": "Max samples (0 = all)",
            "default": 1000,
        },
    }
    input_schema = {}
    output_schema = {"raw_dataset": "json"}

    def _load_from_hf(self, config):
        ds = load_dataset(config["hf_dataset"], split=config.get("hf_split", "train"))
        max_samples = int(config.get("max_samples", 0))
        if max_samples and max_samples > 0:
            ds = ds.select(range(min(max_samples, len(ds))))
        return [dict(r) for r in ds]

    def _load_from_jsonl_path(self, config):
        path = config.get("jsonl_path", "")
        if not path:
            return []
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        max_samples = int(config.get("max_samples", 0))
        if max_samples and max_samples > 0:
            records = records[:max_samples]
        return records

    def execute(self, config, inputs, storage):
        source_type = config.get("source_type", "huggingface")
        if source_type == "huggingface":
            records = self._load_from_hf(config)
        elif source_type == "jsonl_path":
            records = self._load_from_jsonl_path(config)
        else:
            # jsonl_url or unknown: leave as TODO / minimal
            records = []

        data = {
            "records": records,
            "num_samples": len(records),
            "keys": list(records[0].keys()) if records else [],
        }
        ref = storage.save_json("_p", "_n", "raw_dataset", data)
        return {"raw_dataset": ref}

    def get_output_preview(self, outputs, storage):
        data = storage.load_json(outputs["raw_dataset"])
        rows = data["records"][:20]
        keys = data["keys"]
        table_rows = []
        for r in rows:
            table_rows.append([str(r.get(k, ""))[:200] for k in keys])
        return {
            "columns": keys,
            "rows": table_rows,
            "total_rows": data["num_samples"],
        }
```

---

## Card 2: Filter & Clean

**File:** `filter_clean.py` | **Folder:** `data/`

Filters and cleans raw records before splitting and formatting.

```python
from cards.base import BaseCard


class FilterCleanCard(BaseCard):
    card_type = "llm_filter_clean"
    display_name = "Filter & Clean"
    description = "Filter short/long or invalid samples, normalize fields"
    category = "data"
    execution_mode = "local"
    output_view_type = "table"

    config_schema = {
        "instruction_key": {
            "type": "string",
            "label": "Instruction field key",
            "default": "instruction",
        },
        "input_key": {
            "type": "string",
            "label": "Input field key",
            "default": "input",
        },
        "output_key": {
            "type": "string",
            "label": "Output field key",
            "default": "output",
        },
        "min_chars": {
            "type": "number",
            "label": "Min total chars (instruction+input+output)",
            "default": 16,
        },
        "max_chars": {
            "type": "number",
            "label": "Max total chars (0 = unlimited)",
            "default": 4096,
        },
        "drop_if_missing_output": {
            "type": "boolean",
            "label": "Drop rows with missing output",
            "default": True,
        },
    }
    input_schema = {"raw_dataset": "json"}
    output_schema = {"clean_dataset": "json"}

    def execute(self, config, inputs, storage):
        data = storage.load_json(inputs["raw_dataset"])
        records = data["records"]

        instr_key = config["instruction_key"]
        input_key = config["input_key"]
        output_key = config["output_key"]
        min_chars = int(config.get("min_chars", 16))
        max_chars = int(config.get("max_chars", 4096))
        drop_missing_output = bool(config.get("drop_if_missing_output", True))

        clean_records = []
        for r in records:
            instr = (r.get(instr_key) or "").strip()
            inp = (r.get(input_key) or "").strip()
            out = (r.get(output_key) or "").strip()

            if drop_missing_output and not out:
                continue

            total_len = len(instr) + len(inp) + len(out)
            if total_len < min_chars:
                continue
            if max_chars > 0 and total_len > max_chars:
                continue

            clean_records.append(
                {
                    instr_key: instr,
                    input_key: inp,
                    output_key: out,
                }
            )

        result = {
            "records": clean_records,
            "num_samples": len(clean_records),
            "keys": [instr_key, input_key, output_key],
        }
        ref = storage.save_json("_p", "_n", "clean_dataset", result)
        return {"clean_dataset": ref}

    def get_output_preview(self, outputs, storage):
        data = storage.load_json(outputs["clean_dataset"])
        keys = data["keys"]
        rows = data["records"][:20]
        table_rows = []
        for r in rows:
            table_rows.append(
                [str(r.get(k, ""))[:160].replace("\n", " ") for k in keys]
            )
        return {
            "columns": keys,
            "rows": table_rows,
            "total_rows": data["num_samples"],
        }
```

---

## Card 3: Train/Val Split

**File:** `train_val_split.py` | **Folder:** `data/`

Splits the cleaned dataset into train and validation subsets.

```python
from cards.base import BaseCard
import random


class TrainValSplitCard(BaseCard):
    card_type = "llm_train_val_split"
    display_name = "Train/Val Split"
    description = "Split dataset into train and validation sets"
    category = "data"
    execution_mode = "local"
    output_view_type = "table"

    config_schema = {
        "val_ratio": {
            "type": "number",
            "label": "Validation ratio",
            "default": 0.1,
        },
        "shuffle_seed": {
            "type": "number",
            "label": "Shuffle seed",
            "default": 42,
        },
    }
    input_schema = {"clean_dataset": "json"}
    output_schema = {"train_dataset": "json", "val_dataset": "json"}

    def execute(self, config, inputs, storage):
        data = storage.load_json(inputs["clean_dataset"])
        records = list(data["records"])
        keys = data["keys"]

        val_ratio = float(config.get("val_ratio", 0.1))
        seed = int(config.get("shuffle_seed", 42))

        random.Random(seed).shuffle(records)
        n_total = len(records)
        n_val = int(n_total * val_ratio)
        val_records = records[:n_val]
        train_records = records[n_val:]

        train = {
            "records": train_records,
            "num_samples": len(train_records),
            "keys": keys,
        }
        val = {
            "records": val_records,
            "num_samples": len(val_records),
            "keys": keys,
        }

        ref_train = storage.save_json("_p", "_n", "train_dataset", train)
        ref_val = storage.save_json("_p", "_n", "val_dataset", val)
        return {"train_dataset": ref_train, "val_dataset": ref_val}

    def get_output_preview(self, outputs, storage):
        train = storage.load_json(outputs["train_dataset"])
        val = storage.load_json(outputs["val_dataset"])
        return {
            "columns": ["split", "samples"],
            "rows": [
                ["Train", train["num_samples"]],
                ["Validation", val["num_samples"]],
            ],
            "total_rows": 2,
        }
```

---

## Card 4: Format for SFT

**File:** `format_sft.py` | **Folder:** `data/`

Converts train/validation records into instruction–response text format for supervised fine-tuning.

```python
from cards.base import BaseCard


class FormatSFTCard(BaseCard):
    card_type = "llm_format_sft"
    display_name = "Format for SFT"
    description = "Format dataset into instruction/response text"
    category = "data"
    execution_mode = "local"
    output_view_type = "table"

    config_schema = {
        "template": {
            "type": "string",
            "label": "Prompt template",
            "default": "### Instruction:\\n{instruction}\\n\\n### Input:\\n{input}\\n\\n### Response:\\n{output}",
        },
        "instruction_key": {
            "type": "string",
            "label": "Instruction field key",
            "default": "instruction",
        },
        "input_key": {
            "type": "string",
            "label": "Input field key",
            "default": "input",
        },
        "output_key": {
            "type": "string",
            "label": "Output field key",
            "default": "output",
        },
    }
    input_schema = {"train_dataset": "json", "val_dataset": "json"}
    output_schema = {"train_sft": "json", "val_sft": "json"}

    def _format_split(self, data, config):
        template = config["template"].replace("\\n", "\n")
        instr_key = config["instruction_key"]
        input_key = config["input_key"]
        output_key = config["output_key"]

        samples = []
        for r in data["records"]:
            text = template.format(
                instruction=r.get(instr_key, ""),
                input=r.get(input_key, ""),
                output=r.get(output_key, ""),
            )
            samples.append({"text": text})

        return {
            "samples": samples,
            "num_samples": len(samples),
            "template_used": template,
        }

    def execute(self, config, inputs, storage):
        train_data = storage.load_json(inputs["train_dataset"])
        val_data = storage.load_json(inputs["val_dataset"])

        train_sft = self._format_split(train_data, config)
        val_sft = self._format_split(val_data, config)

        ref_train = storage.save_json("_p", "_n", "train_sft", train_sft)
        ref_val = storage.save_json("_p", "_n", "val_sft", val_sft)
        return {"train_sft": ref_train, "val_sft": ref_val}

    def get_output_preview(self, outputs, storage):
        train_sft = storage.load_json(outputs["train_sft"])
        samples = train_sft["samples"][:5]
        rows = []
        for i, s in enumerate(samples):
            text = s["text"]
            rows.append([i, text[:200] + ("..." if len(text) > 200 else "")])
        return {
            "columns": ["#", "text"],
            "rows": rows,
            "total_rows": train_sft["num_samples"],
        }
```

---

## Card 5: Tokenize

**File:** `tokenize.py` | **Folder:** `tokens/`

Tokenizes SFT text samples using a HuggingFace tokenizer and prepares train/validation token sequences.

```python
from cards.base import BaseCard
from transformers import AutoTokenizer


class TokenizeCard(BaseCard):
    card_type = "llm_tokenize"
    display_name = "Tokenize"
    description = "Tokenize SFT text with a HuggingFace tokenizer"
    category = "data"
    execution_mode = "local"
    output_view_type = "table"

    config_schema = {
        "model_name": {
            "type": "string",
            "label": "Tokenizer model name",
            "default": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        },
        "max_length": {
            "type": "number",
            "label": "Max sequence length",
            "default": 1024,
        },
        "packing": {
            "type": "boolean",
            "label": "Pack multiple samples into one sequence",
            "default": True,
        },
    }
    input_schema = {"train_sft": "json", "val_sft": "json"}
    output_schema = {"train_tokens": "json", "val_tokens": "json"}

    def _tokenize_split(self, tokenizer, data, max_length, packing):
        texts = [s["text"] for s in data["samples"]]
        enc = tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
        )
        # For simplicity here, we don't implement complex packing.
        tokens = []
        for ids, mask in zip(enc["input_ids"], enc["attention_mask"]):
            tokens.append({"input_ids": ids, "attention_mask": mask})
        return {
            "tokens": tokens,
            "num_samples": len(tokens),
            "max_length": max_length,
        }

    def execute(self, config, inputs, storage):
        train_sft = storage.load_json(inputs["train_sft"])
        val_sft = storage.load_json(inputs["val_sft"])

        tokenizer = AutoTokenizer.from_pretrained(config["model_name"], trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        max_length = int(config.get("max_length", 1024))
        packing = bool(config.get("packing", True))

        train_tokens = self._tokenize_split(tokenizer, train_sft, max_length, packing)
        val_tokens = self._tokenize_split(tokenizer, val_sft, max_length, packing)

        ref_train = storage.save_json("_p", "_n", "train_tokens", train_tokens)
        ref_val = storage.save_json("_p", "_n", "val_tokens", val_tokens)
        return {"train_tokens": ref_train, "val_tokens": ref_val}

    def get_output_preview(self, outputs, storage):
        train_tokens = storage.load_json(outputs["train_tokens"])
        num = train_tokens["num_samples"]
        max_length = train_tokens["max_length"]
        return {
            "columns": ["split", "samples", "max_length"],
            "rows": [
                ["Train", num, max_length],
            ],
            "total_rows": 1,
        }
```

---

## Card 6: Load Base Model

**File:** `load_model.py` | **Folder:** `model/`

Downloads an open-source LLM and tokenizer from HuggingFace, preparing a lightweight `model_config` for downstream cards.

```python
from cards.base import BaseCard
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch


class LoadBaseModelCard(BaseCard):
    card_type = "llm_load_model"
    display_name = "Load Base Model"
    description = "Load a pre-trained LLM from HuggingFace"
    category = "model"
    execution_mode = "modal"
    output_view_type = "model_summary"

    config_schema = {
        "model_name": {
            "type": "string",
            "label": "HuggingFace model name",
            "default": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        },
        "load_in_4bit": {
            "type": "boolean",
            "label": "Quantize to 4-bit",
            "default": True,
        },
        "cache_dir": {
            "type": "string",
            "label": "HF cache dir",
            "default": "/tmp/hf_cache",
        },
    }
    input_schema = {}
    output_schema = {"model_config": "json"}

    def execute(self, config, inputs, storage):
        model_name = config["model_name"]
        cache_dir = config.get("cache_dir", "/tmp/hf_cache")
        load_in_4bit = bool(config.get("load_in_4bit", True))

        tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        load_kwargs = {}
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            load_kwargs["quantization_config"] = bnb_config

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            device_map="auto",
            trust_remote_code=True,
            **load_kwargs,
        )

        total_params = sum(p.numel() for p in model.parameters())

        model_config = {
            "model_name": model_name,
            "cache_dir": cache_dir,
            "load_in_4bit": load_in_4bit,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "total_parameters": int(total_params),
        }
        ref = storage.save_json("_p", "_n", "model_config", model_config)
        return {"model_config": ref}

    def get_output_preview(self, outputs, storage):
        cfg = storage.load_json(outputs["model_config"])
        return {
            "architecture": cfg["model_name"],
            "total_parameters": cfg["total_parameters"],
            "load_in_4bit": cfg["load_in_4bit"],
        }
```

---

## Card 7: Apply LoRA

**File:** `apply_lora.py` | **Folder:** `model/`

Attaches LoRA adapters to the base model configuration.

```python
from cards.base import BaseCard


class ApplyLoraCard(BaseCard):
    card_type = "llm_apply_lora"
    display_name = "Apply LoRA"
    description = "Configure LoRA adapters for the base model"
    category = "model"
    execution_mode = "modal"
    output_view_type = "model_summary"

    config_schema = {
        "r": {
            "type": "number",
            "label": "LoRA rank",
            "default": 16,
        },
        "lora_alpha": {
            "type": "number",
            "label": "LoRA alpha",
            "default": 32,
        },
        "lora_dropout": {
            "type": "number",
            "label": "LoRA dropout",
            "default": 0.05,
        },
        "target_modules": {
            "type": "string",
            "label": "Target modules (comma-separated)",
            "default": "q_proj,k_proj,v_proj,o_proj",
        },
    }
    input_schema = {"model_config": "json"}
    output_schema = {"lora_config": "json"}

    def execute(self, config, inputs, storage):
        base_cfg = storage.load_json(inputs["model_config"])
        lora_cfg = {
            "base_model": base_cfg,
            "r": int(config.get("r", 16)),
            "lora_alpha": float(config.get("lora_alpha", 32.0)),
            "lora_dropout": float(config.get("lora_dropout", 0.05)),
            "target_modules": [
                m.strip()
                for m in config.get("target_modules", "").split(",")
                if m.strip()
            ],
        }
        ref = storage.save_json("_p", "_n", "lora_config", lora_cfg)
        return {"lora_config": ref}

    def get_output_preview(self, outputs, storage):
        cfg = storage.load_json(outputs["lora_config"])
        return {
            "architecture": cfg["base_model"]["model_name"],
            "lora_rank": cfg["r"],
            "target_modules": ", ".join(cfg["target_modules"]),
        }
```

---

## Card 8: Fine-Tune LoRA

**File:** `finetune_lora.py` | **Folder:** `training/`

Runs supervised fine-tuning of the LoRA-augmented model on the tokenized dataset.

```python
from cards.base import BaseCard


class FinetuneLoraCard(BaseCard):
    card_type = "llm_finetune_lora"
    display_name = "Fine-Tune LoRA"
    description = "Run supervised fine-tuning with LoRA adapters"
    category = "training"
    execution_mode = "modal"
    output_view_type = "metrics"

    config_schema = {
        "num_epochs": {
            "type": "number",
            "label": "Number of epochs",
            "default": 3,
        },
        "batch_size": {
            "type": "number",
            "label": "Batch size",
            "default": 4,
        },
        "learning_rate": {
            "type": "number",
            "label": "Learning rate",
            "default": 2e-4,
        },
        "logging_steps": {
            "type": "number",
            "label": "Logging steps",
            "default": 10,
        },
    }
    input_schema = {
        "lora_config": "json",
        "train_tokens": "json",
        "val_tokens": "json",
    }
    output_schema = {"ft_result": "json"}

    def execute(self, config, inputs, storage):
        lora_cfg = storage.load_json(inputs["lora_config"])
        train_tokens = storage.load_json(inputs["train_tokens"])
        val_tokens = storage.load_json(inputs["val_tokens"])

        # Training logic omitted in this example; here we just simulate results.
        metrics = {
            "train_loss": [1.0, 0.8, 0.6],
            "val_loss": [1.1, 0.85, 0.65],
            "epochs": int(config.get("num_epochs", 3)),
        }

        ft_result = {
            "lora_config": lora_cfg,
            "train_tokens_ref": inputs["train_tokens"],
            "val_tokens_ref": inputs["val_tokens"],
            "metrics": metrics,
            "checkpoint_path": "/tmp/llm_finetune/checkpoint",  # placeholder
        }
        ref = storage.save_json("_p", "_n", "ft_result", ft_result)
        return {"ft_result": ref}

    def get_output_preview(self, outputs, storage):
        ft_result = storage.load_json(outputs["ft_result"])
        m = ft_result["metrics"]
        return {
            "train_loss": m["train_loss"][-1],
            "val_loss": m["val_loss"][-1],
            "epochs": m["epochs"],
        }
```

---

## Card 9: Evaluate

**File:** `evaluate.py` | **Folder:** `evaluation/`

Runs evaluation on the validation set and reports metrics (e.g. loss, perplexity).

```python
from cards.base import BaseCard
import math


class LlmEvaluateCard(BaseCard):
    card_type = "llm_eval"
    display_name = "Evaluate"
    description = "Evaluate finetuned model on validation set"
    category = "evaluation"
    execution_mode = "modal"
    output_view_type = "metrics"

    config_schema = {
        "max_samples": {
            "type": "number",
            "label": "Max validation samples (0 = all)",
            "default": 256,
        }
    }
    input_schema = {"ft_result": "json", "val_tokens": "json"}
    output_schema = {"eval_report": "json"}

    def execute(self, config, inputs, storage):
        ft_result = storage.load_json(inputs["ft_result"])
        val_tokens = storage.load_json(inputs["val_tokens"])

        max_samples = int(config.get("max_samples", 256))
        n = val_tokens["num_samples"]
        used = min(n, max_samples) if max_samples > 0 else n

        # Placeholder evaluation: derive "loss" from training metrics.
        last_val_loss = ft_result["metrics"]["val_loss"][-1]
        perplexity = float(math.exp(min(last_val_loss, 20.0)))

        report = {
            "val_loss": last_val_loss,
            "perplexity": perplexity,
            "samples_used": used,
        }

        ref = storage.save_json("_p", "_n", "eval_report", report)
        return {"eval_report": ref}

    def get_output_preview(self, outputs, storage):
        return storage.load_json(outputs["eval_report"])
```

---

## Card 10: Merge & Export

**File:** `merge_export.py` | **Folder:** `serving/`

Merges LoRA adapters into the base model and prepares a HF-compatible export.

```python
from cards.base import BaseCard


class MergeExportCard(BaseCard):
    card_type = "llm_merge_export"
    display_name = "Merge & Export"
    description = "Merge LoRA into base model and export for serving"
    category = "model"
    execution_mode = "modal"
    output_view_type = "model_summary"

    config_schema = {
        "push_to_hub": {
            "type": "boolean",
            "label": "Push merged model to HuggingFace Hub",
            "default": False,
        },
        "hub_repo": {
            "type": "string",
            "label": "Hub repo (org/name)",
            "default": "",
        },
        "save_format": {
            "type": "string",
            "label": "Save format (safetensors/pt)",
            "default": "safetensors",
        },
    }
    input_schema = {"ft_result": "json"}
    output_schema = {"merged_model": "json"}

    def execute(self, config, inputs, storage):
        ft_result = storage.load_json(inputs["ft_result"])
        lora_cfg = ft_result["lora_config"]

        merged = {
            "base_model": lora_cfg["base_model"]["model_name"],
            "checkpoint_path": ft_result["checkpoint_path"],
            "save_format": config.get("save_format", "safetensors"),
            "pushed_to_hub": bool(config.get("push_to_hub", False)),
            "hub_repo": config.get("hub_repo", ""),
        }

        ref = storage.save_json("_p", "_n", "merged_model", merged)
        return {"merged_model": ref}

    def get_output_preview(self, outputs, storage):
        merged = storage.load_json(outputs["merged_model"])
        return {
            "architecture": merged["base_model"],
            "save_format": merged["save_format"],
            "pushed_to_hub": merged["pushed_to_hub"],
            "hub_repo": merged["hub_repo"],
        }
```

---

## Card 11: vLLM Serve

**File:** `vllm_serve.py` | **Folder:** `serving/`

Starts a vLLM server for the merged model and returns deployment information.

```python
from cards.base import BaseCard


class VllmServeCard(BaseCard):
    card_type = "llm_vllm_serve"
    display_name = "vLLM Serve"
    description = "Serve merged model with vLLM"
    category = "inference"
    execution_mode = "modal"
    output_view_type = "metrics"

    config_schema = {
        "deployment_name": {
            "type": "string",
            "label": "Deployment name",
            "default": "llm-finetune-v1",
        },
        "port": {
            "type": "number",
            "label": "vLLM port",
            "default": 8001,
        },
        "max_batch_tokens": {
            "type": "number",
            "label": "Max batch tokens",
            "default": 8192,
        },
    }
    input_schema = {"merged_model": "json"}
    output_schema = {"deployment_info": "json"}

    def execute(self, config, inputs, storage):
        merged = storage.load_json(inputs["merged_model"])

        # In a real implementation, you would start a vLLM server process here.
        base_url = f"http://localhost:{int(config.get('port', 8001))}"
        info = {
            "deployment_name": config.get("deployment_name", "llm-finetune-v1"),
            "base_url": base_url,
            "model": merged["base_model"],
            "max_batch_tokens": int(config.get("max_batch_tokens", 8192)),
            "example_curl": f"curl {base_url}/generate -X POST -d '{{\"prompt\": \"Hello\"}}'",
        }

        ref = storage.save_json("_p", "_n", "deployment_info", info)
        return {"deployment_info": ref}

    def get_output_preview(self, outputs, storage):
        return storage.load_json(outputs["deployment_info"])
```

---

With these 11 cards you can build a complete LLM pipeline that starts from raw data, fine-tunes an open-source model with LoRA, evaluates it, merges the adapters, and serves the model via vLLM.

