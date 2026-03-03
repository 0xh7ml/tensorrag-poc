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

        tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

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
