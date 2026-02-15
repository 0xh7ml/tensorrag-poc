/* eslint-disable @typescript-eslint/no-explicit-any */

/**
 * Registers BaseCard-aware Python completions and hover hints for Monaco.
 */
export function setupBaseCardIntelliSense(monaco: any): void {
  // Avoid duplicate registration
  const providerId = "__tensorrag_python_completion";
  if ((monaco as any)[providerId]) return;
  (monaco as any)[providerId] = true;

  monaco.languages.registerCompletionItemProvider("python", {
    triggerCharacters: [".", '"'],
    provideCompletionItems(_model: any, position: any) {
      const range = {
        startLineNumber: position.lineNumber,
        startColumn: position.column,
        endLineNumber: position.lineNumber,
        endColumn: position.column,
      };

      const suggestions = [
        // Class attributes
        {
          label: "card_type",
          kind: monaco.languages.CompletionItemKind.Property,
          insertText: 'card_type = "${1:my_card}"',
          insertTextRules:
            monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
          detail: "Unique card identifier",
          documentation: "A unique snake_case identifier for this card type.",
          range,
        },
        {
          label: "display_name",
          kind: monaco.languages.CompletionItemKind.Property,
          insertText: 'display_name = "${1:My Card}"',
          insertTextRules:
            monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
          detail: "UI display name",
          documentation: "The human-readable name shown in the card palette.",
          range,
        },
        {
          label: "description",
          kind: monaco.languages.CompletionItemKind.Property,
          insertText: 'description = "${1:What this card does}"',
          insertTextRules:
            monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
          detail: "Card description",
          range,
        },
        {
          label: "category",
          kind: monaco.languages.CompletionItemKind.Property,
          insertText: 'category = "${1|data,model,evaluation,inference|}"',
          insertTextRules:
            monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
          detail: "Card category",
          documentation:
            "Card category: data, model, evaluation, or inference.",
          range,
        },
        {
          label: "execution_mode",
          kind: monaco.languages.CompletionItemKind.Property,
          insertText: 'execution_mode = "${1|local,modal|}"',
          insertTextRules:
            monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
          detail: "Execution mode",
          documentation: "local = CPU, modal = remote GPU.",
          range,
        },
        {
          label: "output_view_type",
          kind: monaco.languages.CompletionItemKind.Property,
          insertText: 'output_view_type = "${1|table,metrics,model_summary|}"',
          insertTextRules:
            monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
          detail: "Output view type",
          documentation: "How the output is displayed in the UI.",
          range,
        },
        {
          label: "config_schema",
          kind: monaco.languages.CompletionItemKind.Property,
          insertText:
            'config_schema = {\n\t"type": "object",\n\t"properties": {\n\t\t"${1:param}": {\n\t\t\t"type": "${2:string}",\n\t\t\t"description": "${3:Description}"\n\t\t}\n\t},\n\t"required": []\n}',
          insertTextRules:
            monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
          detail: "JSON Schema for config",
          documentation: "Defines the configuration form shown in the UI.",
          range,
        },
        {
          label: "input_schema",
          kind: monaco.languages.CompletionItemKind.Property,
          insertText: 'input_schema = {"${1:input_name}": "${2:dataframe}"}',
          insertTextRules:
            monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
          detail: "Input ports",
          documentation:
            'Dict of input port names to types (dataframe, json, model, bytes).',
          range,
        },
        {
          label: "output_schema",
          kind: monaco.languages.CompletionItemKind.Property,
          insertText: 'output_schema = {"${1:output_name}": "${2:dataframe}"}',
          insertTextRules:
            monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
          detail: "Output ports",
          documentation:
            'Dict of output port names to types (dataframe, json, model, bytes).',
          range,
        },

        // Methods
        {
          label: "def execute",
          kind: monaco.languages.CompletionItemKind.Method,
          insertText:
            'def execute(self, config: dict, inputs: dict, storage) -> dict:\n\t"""Run the card logic."""\n\tpid = config["_pipeline_id"]\n\tnid = config["_node_id"]\n\t${0}',
          insertTextRules:
            monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
          detail: "Card execution method",
          documentation:
            "Required. Receives config dict, inputs dict (storage refs), and storage service. Must return dict of output key -> storage ref.",
          range,
        },
        {
          label: "def get_output_preview",
          kind: monaco.languages.CompletionItemKind.Method,
          insertText:
            'def get_output_preview(self, outputs: dict, storage) -> dict:\n\t"""Return a frontend-friendly preview."""\n\t${0}',
          insertTextRules:
            monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
          detail: "Output preview method",
          documentation:
            "Required. Receives outputs dict (storage refs) and storage service. Must return JSON-serializable dict.",
          range,
        },

        // Storage API
        {
          label: "storage.save_dataframe",
          kind: monaco.languages.CompletionItemKind.Function,
          insertText:
            'storage.save_dataframe(pid, nid, "${1:key}", ${2:df})',
          insertTextRules:
            monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
          detail: "Save DataFrame to storage",
          documentation: "Saves a pandas DataFrame as parquet. Returns storage reference.",
          range,
        },
        {
          label: "storage.load_dataframe",
          kind: monaco.languages.CompletionItemKind.Function,
          insertText:
            'storage.load_dataframe(${1:ref})',
          insertTextRules:
            monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
          detail: "Load DataFrame from storage",
          documentation: "Loads a pandas DataFrame from a storage reference.",
          range,
        },
        {
          label: "storage.save_json",
          kind: monaco.languages.CompletionItemKind.Function,
          insertText:
            'storage.save_json(pid, nid, "${1:key}", ${2:data})',
          insertTextRules:
            monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
          detail: "Save JSON to storage",
          range,
        },
        {
          label: "storage.load_json",
          kind: monaco.languages.CompletionItemKind.Function,
          insertText: "storage.load_json(${1:ref})",
          insertTextRules:
            monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
          detail: "Load JSON from storage",
          range,
        },
        {
          label: "storage.save_model",
          kind: monaco.languages.CompletionItemKind.Function,
          insertText:
            'storage.save_model(pid, nid, "${1:key}", ${2:model})',
          insertTextRules:
            monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
          detail: "Save model (joblib) to storage",
          range,
        },
        {
          label: "storage.load_model",
          kind: monaco.languages.CompletionItemKind.Function,
          insertText: "storage.load_model(${1:ref})",
          insertTextRules:
            monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
          detail: "Load model (joblib) from storage",
          range,
        },
      ];

      return { suggestions };
    },
  });
}
