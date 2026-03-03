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
