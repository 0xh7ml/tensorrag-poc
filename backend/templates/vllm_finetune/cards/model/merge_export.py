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

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

        model = PeftModel.from_pretrained(base_model, adapter_dir)

        model = model.merge_and_unload()

        model.save_pretrained(merged_dir)

        tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
        tokenizer.save_pretrained(merged_dir)

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
            "status": "Merge complete - ready for vLLM",
        }
