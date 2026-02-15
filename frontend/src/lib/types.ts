// --- JSON Schema subset for config rendering ---
export interface JsonSchemaProperty {
  type: string;
  description?: string;
  enum?: (string | number)[];
  default?: unknown;
  items?: { type: string; enum?: string[] };
}

export interface JsonSchema {
  type: string;
  properties?: Record<string, JsonSchemaProperty>;
  required?: string[];
}

// --- Card Schema (from GET /api/cards) ---
export interface CardSchema {
  card_type: string;
  display_name: string;
  description: string;
  category: "data" | "model" | "evaluation" | "inference";
  execution_mode: "local" | "modal";
  config_schema: JsonSchema;
  input_schema: Record<string, string>;
  output_schema: Record<string, string>;
  output_view_type: "table" | "metrics" | "model_summary";
}

// --- Pipeline models ---
export interface NodeConfig {
  id: string;
  type: string;
  config: Record<string, unknown>;
  position?: { x: number; y: number };
}

export interface EdgeConfig {
  source: string;
  target: string;
  source_output: string;
  target_input: string;
}

export interface PipelineRequest {
  pipeline_id: string;
  nodes: NodeConfig[];
  edges: EdgeConfig[];
}

// --- Status ---
export type NodeStatusValue = "pending" | "running" | "completed" | "failed";

export interface NodeStatus {
  node_id: string;
  status: NodeStatusValue;
  error: string | null;
}

export interface PipelineStatus {
  pipeline_id: string;
  status: string;
  node_statuses: Record<string, NodeStatus>;
}

// --- Output preview ---
export interface CardOutputPreview {
  node_id: string;
  output_type: string;
  preview: Record<string, unknown>;
}

// --- WebSocket messages ---
export interface WSNodeStatusMessage {
  type: "node_status";
  node_id: string;
  status: NodeStatusValue;
  message: string;
  timestamp: string;
}

export interface WSLogMessage {
  type: "log";
  text: string;
  timestamp: string;
}

export type WSMessage = WSNodeStatusMessage | WSLogMessage;

// --- Custom card editor types ---
export interface CardValidationError {
  line: number | null;
  message: string;
  severity: "error" | "warning";
}

export interface CardValidationResult {
  success: boolean;
  errors: CardValidationError[];
  extracted_schema: CardSchema | null;
}

export interface CustomCardFile {
  filename: string;
  source_code: string;
  card_type: string;
}

// --- React Flow node data ---
export interface CardNodeData {
  label: string;
  cardSchema: CardSchema;
  config: Record<string, unknown>;
  status: NodeStatusValue;
  error: string | null;
  outputPreview: CardOutputPreview | null;
  [key: string]: unknown;
}
