function toPascalCase(str: string): string {
  return str
    .split("_")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join("");
}

export function generateCardTemplate(
  cardType: string,
  displayName: string
): string {
  const className = `${toPascalCase(cardType)}Card`;

  return `from cards.base import BaseCard


class ${className}(BaseCard):
    card_type = "${cardType}"
    display_name = "${displayName}"
    description = "Custom card: ${displayName}"
    category = "data"  # data | model | evaluation | inference
    execution_mode = "local"  # local | modal
    output_view_type = "table"  # table | metrics | model_summary

    config_schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }
    input_schema = {}
    output_schema = {}

    def execute(self, config: dict, inputs: dict, storage) -> dict:
        \"\"\"Run the card logic. Returns dict of output key -> storage ref.\"\"\"
        pid = config["_pipeline_id"]
        nid = config["_node_id"]

        # Example: load an input
        # df = storage.load_dataframe(inputs["dataset"])

        # Example: process data
        # result = df.copy()

        # Example: save an output
        # ref = storage.save_dataframe(pid, nid, "output_key", result)
        # return {"output_key": ref}

        raise NotImplementedError("Implement execute()")

    def get_output_preview(self, outputs: dict, storage) -> dict:
        \"\"\"Return a frontend-friendly preview of the card's output.\"\"\"

        # Example: load output and return preview
        # df = storage.load_dataframe(outputs["output_key"])
        # return {
        #     "rows": df.head(50).to_dict(orient="records"),
        #     "columns": [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns],
        #     "shape": {"rows": len(df), "cols": len(df.columns)},
        # }

        raise NotImplementedError("Implement get_output_preview()")
`;
}
