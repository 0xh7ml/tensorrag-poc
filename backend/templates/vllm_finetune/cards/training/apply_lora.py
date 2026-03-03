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
