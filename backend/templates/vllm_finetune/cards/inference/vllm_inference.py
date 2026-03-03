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

        raw = config.get("prompts", "").replace("\\n", "\n")
        prompts = [p.strip() for p in raw.split("---") if p.strip()]
        if not prompts:
            prompts = [raw]

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
