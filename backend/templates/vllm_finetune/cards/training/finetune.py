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

        tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

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

        peft_config = LoraConfig(
            r=lora_cfg["lora_r"],
            lora_alpha=lora_cfg["lora_alpha"],
            lora_dropout=lora_cfg["lora_dropout"],
            target_modules=lora_cfg["target_modules"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)

        train_ds = Dataset.from_list(sft_data["samples"])

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

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_ds,
            args=training_args,
            tokenizer=tokenizer,
            max_seq_length=int(config.get("max_seq_length", 512)),
        )

        result = trainer.train()

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
