# vLLM Fine-Tuning & Inference Pipeline

A visual pipeline for LLM fine-tuning with LoRA and fast inference with vLLM — broken into 7 minimal cards.

## Pipeline Overview

```
Load Dataset → Format for SFT → Load Base Model → Apply LoRA → Fine-Tune → Merge & Export → vLLM Inference
```

## Connection Map

| # | Card | Input | Output | Purpose |
|---|------|-------|--------|---------|
| 1 | Load Dataset | — | `dataset` | Load raw data from HuggingFace or file |
| 2 | Format for SFT | `dataset` | `sft_dataset` | Convert to instruction/response format |
| 3 | Load Base Model | — | `model_config` | Download pre-trained LLM + tokenizer |
| 4 | Apply LoRA | `model_config` | `lora_config` | Attach LoRA adapters to the model |
| 5 | Fine-Tune | `lora_config`, `sft_dataset` | `training_result` | Run supervised fine-tuning |
| 6 | Merge & Export | `training_result` | `merged_model` | Merge LoRA into base model |
| 7 | vLLM Inference | `merged_model` | `generations` | Fast batched inference |

## Wiring

```
[Load Dataset] ──dataset──> [Format for SFT]
[Format for SFT] ──sft_dataset──> [Fine-Tune]
[Load Base Model] ──model_config──> [Apply LoRA]
[Apply LoRA] ──lora_config──> [Fine-Tune]
[Fine-Tune] ──training_result──> [Merge & Export]
[Merge & Export] ──merged_model──> [vLLM Inference]
```

## Requirements

```bash
pip install transformers datasets peft trl accelerate bitsandbytes vllm torch
```

> **Note:** Fine-tuning and vLLM inference require GPU. Use `execution_mode = "modal"` to run on cloud GPUs, or `"local"` if you have a GPU machine.

---

## Card 1: Load Dataset

Loads a dataset from HuggingFace Hub or a JSON/CSV file.

```python
from cards.base import BaseCard

class LoadDatasetCard(BaseCard):
    card_type = "llm_load_dataset"
    display_name = "Load Dataset"
    description = "Load dataset from HuggingFace or file"
    category = "data"
    execution_mode = "local"
    output_view_type = "table"

    config_schema = {
        "source": {
            "type": "string",
            "label": "HuggingFace dataset name or file URL",
            "default": "tatsu-lab/alpaca"
        },
        "split": {
            "type": "string",
            "label": "Split",
            "default": "train"
        },
        "max_samples": {
            "type": "number",
            "label": "Max samples (0 = all)",
            "default": 1000
        }
    }
    input_schema = {}
    output_schema = {"dataset": "json"}

    def execute(self, config, inputs, storage):
        from datasets import load_dataset

        source = config["source"]
        split = config.get("split", "train")
        max_samples = int(config.get("max_samples", 1000))

        ds = load_dataset(source, split=split)

        if max_samples > 0:
            ds = ds.select(range(min(max_samples, len(ds))))

        # Store as list of dicts
        records = [dict(row) for row in ds]

        data = {
            "records": records,
            "num_samples": len(records),
            "columns": list(records[0].keys()) if records else [],
            "source": source,
        }
        ref = storage.save_json("_p", "_n", "dataset", data)
        return {"dataset": ref}

    def get_output_preview(self, outputs, storage):
        data = storage.load_json(outputs["dataset"])
        records = data["records"][:10]
        columns = data["columns"]

        # Truncate long text for preview
        rows = []
        for r in records:
            row = []
            for c in columns:
                val = str(r.get(c, ""))
                row.append(val[:100] + "..." if len(val) > 100 else val)
            rows.append(row)

        return {
            "columns": columns,
            "rows": rows,
            "total_rows": data["num_samples"],
        }
```

---

## Card 2: Format for SFT

Converts raw dataset into the instruction-response format needed for supervised fine-tuning.

```python
from cards.base import BaseCard

class FormatSFTCard(BaseCard):
    card_type = "llm_format_sft"
    display_name = "Format for SFT"
    description = "Format dataset for supervised fine-tuning"
    category = "data"
    execution_mode = "local"
    output_view_type = "table"

    config_schema = {
        "template": {
            "type": "string",
            "label": "Prompt template",
            "default": "### Instruction:\\n{instruction}\\n\\n### Input:\\n{input}\\n\\n### Response:\\n{output}"
        },
        "instruction_col": {
            "type": "string",
            "label": "Instruction column",
            "default": "instruction"
        },
        "input_col": {
            "type": "string",
            "label": "Input column (optional context)",
            "default": "input"
        },
        "output_col": {
            "type": "string",
            "label": "Output/response column",
            "default": "output"
        }
    }
    input_schema = {"dataset": "json"}
    output_schema = {"sft_dataset": "json"}

    def execute(self, config, inputs, storage):
        data = storage.load_json(inputs["dataset"])
        records = data["records"]

        template = config["template"].replace("\\n", "\n")
        instr_col = config["instruction_col"]
        input_col = config["input_col"]
        output_col = config["output_col"]

        formatted = []
        for r in records:
            text = template.format(
                instruction=r.get(instr_col, ""),
                input=r.get(input_col, ""),
                output=r.get(output_col, ""),
            )
            formatted.append({"text": text})

        sft_data = {
            "samples": formatted,
            "num_samples": len(formatted),
            "template_used": template,
        }
        ref = storage.save_json("_p", "_n", "sft_dataset", sft_data)
        return {"sft_dataset": ref}

    def get_output_preview(self, outputs, storage):
        data = storage.load_json(outputs["sft_dataset"])
        samples = data["samples"][:5]
        rows = []
        for i, s in enumerate(samples):
            text = s["text"]
            rows.append([i, text[:200] + "..." if len(text) > 200 else text])

        return {
            "columns": ["#", "formatted_text"],
            "rows": rows,
            "total_rows": data["num_samples"],
        }
```

---

## Card 3: Load Base Model

Downloads a pre-trained LLM and its tokenizer from HuggingFace.

```python
from cards.base import BaseCard

class LoadBaseModelCard(BaseCard):
    card_type = "llm_load_model"
    display_name = "Load Base Model"
    description = "Load a pre-trained LLM from HuggingFace"
    category = "model"
    execution_mode = "local"
    output_view_type = "model_summary"

    config_schema = {
        "model_name": {
            "type": "string",
            "label": "HuggingFace model name",
            "default": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        },
        "load_in_4bit": {
            "type": "boolean",
            "label": "Quantize to 4-bit (saves GPU memory)",
            "default": True
        },
        "cache_dir": {
            "type": "string",
            "label": "Cache directory",
            "default": "/tmp/hf_cache"
        }
    }
    input_schema = {}
    output_schema = {"model_config": "json"}

    def execute(self, config, inputs, storage):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch

        model_name = config["model_name"]
        cache_dir = config.get("cache_dir", "/tmp/hf_cache")
        load_in_4bit = config.get("load_in_4bit", True)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model (optionally quantized)
        load_kwargs = {
            "cache_dir": cache_dir,
            "device_map": "auto",
            "trust_remote_code": True,
        }
        if load_in_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        model_config = {
            "model_name": model_name,
            "cache_dir": cache_dir,
            "load_in_4bit": load_in_4bit,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "vocab_size": len(tokenizer),
            "model_type": model.config.model_type,
            "hidden_size": model.config.hidden_size,
            "num_layers": model.config.num_hidden_layers,
            "num_heads": model.config.num_attention_heads,
        }
        ref = storage.save_json("_p", "_n", "model_config", model_config)
        return {"model_config": ref}

    def get_output_preview(self, outputs, storage):
        cfg = storage.load_json(outputs["model_config"])
        return {
            "model_name": cfg["model_name"],
            "model_type": cfg["model_type"],
            "total_parameters": f"{cfg['total_params']:,}",
            "hidden_size": cfg["hidden_size"],
            "layers": cfg["num_layers"],
            "attention_heads": cfg["num_heads"],
            "quantized_4bit": cfg["load_in_4bit"],
        }
```

---

## Card 4: Apply LoRA

Attaches LoRA (Low-Rank Adaptation) adapters to the base model. Only a small fraction of parameters become trainable.

```python
from cards.base import BaseCard

class ApplyLoRACard(BaseCard):
    card_type = "llm_apply_lora"
    display_name = "Apply LoRA"
    description = "Attach LoRA adapters to the model"
    category = "training"
    execution_mode = "local"
    output_view_type = "model_summary"

    config_schema = {
        "r": {
            "type": "number",
            "label": "LoRA rank (r)",
            "default": 16
        },
        "alpha": {
            "type": "number",
            "label": "LoRA alpha",
            "default": 32
        },
        "dropout": {
            "type": "number",
            "label": "LoRA dropout",
            "default": 0.05
        },
        "target_modules": {
            "type": "string",
            "label": "Target modules (comma-separated)",
            "default": "q_proj,v_proj,k_proj,o_proj"
        }
    }
    input_schema = {"model_config": "json"}
    output_schema = {"lora_config": "json"}

    def execute(self, config, inputs, storage):
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        import torch

        model_cfg = storage.load_json(inputs["model_config"])
        model_name = model_cfg["model_name"]
        cache_dir = model_cfg["cache_dir"]

        # Reload model
        load_kwargs = {"cache_dir": cache_dir, "device_map": "auto", "trust_remote_code": True}
        if model_cfg.get("load_in_4bit"):
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

        if model_cfg.get("load_in_4bit"):
            model = prepare_model_for_kbit_training(model)

        # Apply LoRA
        target_modules = [m.strip() for m in config["target_modules"].split(",")]
        lora_cfg = LoraConfig(
            r=int(config["r"]),
            lora_alpha=int(config["alpha"]),
            lora_dropout=float(config["dropout"]),
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

        peft_model = get_peft_model(model, lora_cfg)

        trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in peft_model.parameters())
        pct = round(100 * trainable / total, 2) if total > 0 else 0

        lora_result = {
            **model_cfg,
            "lora_r": int(config["r"]),
            "lora_alpha": int(config["alpha"]),
            "lora_dropout": float(config["dropout"]),
            "target_modules": target_modules,
            "trainable_params": trainable,
            "total_params": total,
            "trainable_pct": pct,
        }
        ref = storage.save_json("_p", "_n", "lora_config", lora_result)
        return {"lora_config": ref}

    def get_output_preview(self, outputs, storage):
        cfg = storage.load_json(outputs["lora_config"])
        return {
            "model": cfg["model_name"],
            "lora_rank": cfg["lora_r"],
            "lora_alpha": cfg["lora_alpha"],
            "target_modules": ", ".join(cfg["target_modules"]),
            "trainable_params": f"{cfg['trainable_params']:,}",
            "total_params": f"{cfg['total_params']:,}",
            "trainable_pct": f"{cfg['trainable_pct']}%",
        }
```

---

## Card 5: Fine-Tune

Runs supervised fine-tuning (SFT) using HuggingFace's `trl.SFTTrainer`.

```python
from cards.base import BaseCard

class FineTuneCard(BaseCard):
    card_type = "llm_finetune"
    display_name = "Fine-Tune"
    description = "Supervised fine-tuning with SFTTrainer"
    category = "training"
    execution_mode = "local"
    output_view_type = "metrics"

    config_schema = {
        "epochs": {
            "type": "number",
            "label": "Number of epochs",
            "default": 3
        },
        "batch_size": {
            "type": "number",
            "label": "Per-device batch size",
            "default": 4
        },
        "learning_rate": {
            "type": "number",
            "label": "Learning rate",
            "default": 0.0002
        },
        "max_seq_length": {
            "type": "number",
            "label": "Max sequence length",
            "default": 512
        },
        "output_dir": {
            "type": "string",
            "label": "Adapter output directory",
            "default": "/tmp/lora_adapters"
        }
    }
    input_schema = {"lora_config": "json", "sft_dataset": "json"}
    output_schema = {"training_result": "json"}

    def execute(self, config, inputs, storage):
        from transformers import (
            AutoModelForCausalLM, AutoTokenizer,
            TrainingArguments, BitsAndBytesConfig,
        )
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from trl import SFTTrainer
        from datasets import Dataset
        import torch

        lora_cfg = storage.load_json(inputs["lora_config"])
        sft_data = storage.load_json(inputs["sft_dataset"])

        model_name = lora_cfg["model_name"]
        cache_dir = lora_cfg["cache_dir"]
        output_dir = config.get("output_dir", "/tmp/lora_adapters")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        load_kwargs = {"cache_dir": cache_dir, "device_map": "auto", "trust_remote_code": True}
        if lora_cfg.get("load_in_4bit"):
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

        if lora_cfg.get("load_in_4bit"):
            model = prepare_model_for_kbit_training(model)

        # Apply LoRA
        peft_config = LoraConfig(
            r=lora_cfg["lora_r"],
            lora_alpha=lora_cfg["lora_alpha"],
            lora_dropout=lora_cfg["lora_dropout"],
            target_modules=lora_cfg["target_modules"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)

        # Prepare dataset
        train_ds = Dataset.from_list(sft_data["samples"])

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=int(config.get("epochs", 3)),
            per_device_train_batch_size=int(config.get("batch_size", 4)),
            learning_rate=float(config.get("learning_rate", 2e-4)),
            logging_steps=10,
            save_strategy="epoch",
            fp16=True,
            gradient_accumulation_steps=4,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            report_to="none",
        )

        # Train
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_ds,
            args=training_args,
            tokenizer=tokenizer,
            max_seq_length=int(config.get("max_seq_length", 512)),
        )

        result = trainer.train()

        # Save adapter
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        training_result = {
            "model_name": model_name,
            "cache_dir": cache_dir,
            "adapter_dir": output_dir,
            "load_in_4bit": lora_cfg.get("load_in_4bit", False),
            "lora_r": lora_cfg["lora_r"],
            "lora_alpha": lora_cfg["lora_alpha"],
            "epochs": int(config.get("epochs", 3)),
            "train_loss": round(result.training_loss, 4),
            "train_samples": sft_data["num_samples"],
            "train_runtime_sec": round(result.metrics.get("train_runtime", 0), 1),
        }
        ref = storage.save_json("_p", "_n", "training_result", training_result)
        return {"training_result": ref}

    def get_output_preview(self, outputs, storage):
        result = storage.load_json(outputs["training_result"])
        return {
            "model": result["model_name"],
            "epochs": result["epochs"],
            "training_loss": result["train_loss"],
            "train_samples": result["train_samples"],
            "runtime_seconds": result["train_runtime_sec"],
            "adapter_saved_to": result["adapter_dir"],
        }
```

---

## Card 6: Merge & Export

Merges LoRA adapter weights back into the base model and saves the full merged model to disk.

```python
from cards.base import BaseCard

class MergeExportCard(BaseCard):
    card_type = "llm_merge_export"
    display_name = "Merge & Export"
    description = "Merge LoRA adapters into base model"
    category = "model"
    execution_mode = "local"
    output_view_type = "metrics"

    config_schema = {
        "merged_output_dir": {
            "type": "string",
            "label": "Merged model output directory",
            "default": "/tmp/merged_model"
        }
    }
    input_schema = {"training_result": "json"}
    output_schema = {"merged_model": "json"}

    def execute(self, config, inputs, storage):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        import torch
        import os

        result = storage.load_json(inputs["training_result"])
        model_name = result["model_name"]
        adapter_dir = result["adapter_dir"]
        cache_dir = result["cache_dir"]
        merged_dir = config.get("merged_output_dir", "/tmp/merged_model")

        # Load base model (full precision for merging)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

        # Load adapter on top
        model = PeftModel.from_pretrained(base_model, adapter_dir)

        # Merge and unload adapter
        model = model.merge_and_unload()

        # Save merged model
        model.save_pretrained(merged_dir)

        tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
        tokenizer.save_pretrained(merged_dir)

        # Get model size
        total_size = sum(
            os.path.getsize(os.path.join(merged_dir, f))
            for f in os.listdir(merged_dir)
            if f.endswith((".safetensors", ".bin"))
        )

        merged_info = {
            "merged_model_dir": merged_dir,
            "base_model_name": model_name,
            "lora_r": result["lora_r"],
            "model_size_mb": round(total_size / 1024 / 1024, 1),
        }
        ref = storage.save_json("_p", "_n", "merged_model", merged_info)
        return {"merged_model": ref}

    def get_output_preview(self, outputs, storage):
        info = storage.load_json(outputs["merged_model"])
        return {
            "base_model": info["base_model_name"],
            "lora_rank_used": info["lora_r"],
            "merged_size_mb": info["model_size_mb"],
            "saved_to": info["merged_model_dir"],
            "status": "Merge complete — ready for vLLM",
        }
```

---

## Card 7: vLLM Inference

Loads the merged model with vLLM for fast batched inference.

```python
from cards.base import BaseCard

class VLLMInferenceCard(BaseCard):
    card_type = "llm_vllm_inference"
    display_name = "vLLM Inference"
    description = "Fast inference with vLLM"
    category = "inference"
    execution_mode = "local"
    output_view_type = "table"

    config_schema = {
        "prompts": {
            "type": "string",
            "label": "Prompts (one per line)",
            "default": "### Instruction:\\nExplain what machine learning is in one sentence.\\n\\n### Response:\\n"
        },
        "max_tokens": {
            "type": "number",
            "label": "Max new tokens",
            "default": 256
        },
        "temperature": {
            "type": "number",
            "label": "Temperature",
            "default": 0.7
        },
        "top_p": {
            "type": "number",
            "label": "Top-p (nucleus sampling)",
            "default": 0.9
        }
    }
    input_schema = {"merged_model": "json"}
    output_schema = {"generations": "json"}

    def execute(self, config, inputs, storage):
        from vllm import LLM, SamplingParams

        merged = storage.load_json(inputs["merged_model"])
        model_dir = merged["merged_model_dir"]

        # Parse prompts
        raw = config.get("prompts", "").replace("\\n", "\n")
        prompts = [p.strip() for p in raw.split("---") if p.strip()]
        if not prompts:
            prompts = [raw]

        # vLLM engine
        llm = LLM(model=model_dir, trust_remote_code=True)

        sampling = SamplingParams(
            max_tokens=int(config.get("max_tokens", 256)),
            temperature=float(config.get("temperature", 0.7)),
            top_p=float(config.get("top_p", 0.9)),
        )

        outputs = llm.generate(prompts, sampling)

        generations = []
        for i, out in enumerate(outputs):
            generated_text = out.outputs[0].text
            generations.append({
                "prompt": prompts[i][:200],
                "response": generated_text,
                "tokens_generated": len(out.outputs[0].token_ids),
            })

        total_tokens = sum(g["tokens_generated"] for g in generations)
        result = {
            "generations": generations,
            "num_prompts": len(prompts),
            "total_tokens_generated": total_tokens,
            "model_dir": model_dir,
        }
        ref = storage.save_json("_p", "_n", "generations", result)
        return {"generations": ref}

    def get_output_preview(self, outputs, storage):
        data = storage.load_json(outputs["generations"])
        rows = []
        for g in data["generations"]:
            prompt = g["prompt"][:80] + "..." if len(g["prompt"]) > 80 else g["prompt"]
            response = g["response"][:200] + "..." if len(g["response"]) > 200 else g["response"]
            rows.append([prompt, response, g["tokens_generated"]])

        return {
            "columns": ["prompt", "response", "tokens"],
            "rows": rows,
            "total_rows": data["num_prompts"],
        }
```

---

## Quick Start Example

### 1. Create a project

In the Board view sidebar, click **+ New Project** and name it (e.g. `vllm-finetune`). All cards and canvas state will be saved under this project.

### 2. Create the cards

Switch to the **Editor** view. Create each card file — you can organize them in folders (e.g. `data/`, `training/`, `inference/`). Paste the code, validate, and **Publish to Board** each one.

### 3. Wire the pipeline on the canvas

```
[Load Dataset]  ─────────────────────────────> [Format for SFT]
                                                       │
[Load Base Model] ──> [Apply LoRA]                     │
                           │                           │
                           └──────> [Fine-Tune] <──────┘
                                        │
                                  [Merge & Export]
                                        │
                                  [vLLM Inference]
```

### 4. Configure

| Card | Key Settings |
|------|-------------|
| Load Dataset | `tatsu-lab/alpaca`, max 1000 samples |
| Format for SFT | Alpaca template (default works) |
| Load Base Model | `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, 4-bit on |
| Apply LoRA | r=16, alpha=32, target `q_proj,v_proj,k_proj,o_proj` |
| Fine-Tune | 3 epochs, batch_size 4, lr 2e-4 |
| Merge & Export | defaults |
| vLLM Inference | Type your prompt |

### 5. Run

Click **Run** — the pipeline executes left-to-right. Click any card to see its output.

> **Note:** Your pipeline canvas is automatically saved to S3 per-project. Switching projects in the sidebar loads that project's canvas and cards. You can come back to this pipeline anytime by selecting the project.

---

## Recommended Models for Testing

| Model | Size | Notes |
|-------|------|-------|
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 1.1B | Fast, good for testing |
| `microsoft/phi-2` | 2.7B | Strong for its size |
| `mistralai/Mistral-7B-v0.1` | 7B | Needs ~6GB VRAM (4-bit) |
| `meta-llama/Llama-2-7b-hf` | 7B | Requires HF access token |

## Key Concepts

- **LoRA** (Low-Rank Adaptation): Instead of training all billions of parameters, LoRA freezes the base model and injects small trainable matrices into attention layers. This cuts memory usage by ~90% and trains 10x faster.

- **QLoRA** (4-bit LoRA): Loads the base model in 4-bit precision using `bitsandbytes`, reducing VRAM usage further. A 7B model fits in ~6GB VRAM.

- **SFTTrainer**: HuggingFace's `trl` library wraps the standard Trainer with SFT-specific features like packing short examples together and handling chat templates.

- **Merging**: After training, LoRA adapters are separate files (~50MB). Merging folds them back into the base model so you get a single self-contained model.

- **vLLM**: A high-throughput inference engine that uses PagedAttention for efficient memory management. It's 2-4x faster than standard HuggingFace generation for batched requests.
