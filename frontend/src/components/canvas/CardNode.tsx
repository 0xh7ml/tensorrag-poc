"use client";

import { memo } from "react";
import { Handle, Position, type NodeProps, type Node } from "@xyflow/react";
import type { CardNodeData } from "@/lib/types";
import { usePipelineStore } from "@/store/pipelineStore";
import { runPipeline } from "@/lib/pipelineRunner";

const CATEGORY_COLORS: Record<string, { border: string; bg: string }> = {
  data: { border: "border-card-data/50", bg: "bg-card-data/5" },
  model: { border: "border-card-model/50", bg: "bg-card-model/5" },
  evaluation: { border: "border-card-evaluation/50", bg: "bg-card-evaluation/5" },
  inference: { border: "border-card-inference/50", bg: "bg-card-inference/5" },
  training: { border: "border-card-model/50", bg: "bg-card-model/5" },
};

const STATUS_DOT: Record<string, string> = {
  pending: "bg-status-pending",
  running: "bg-status-running animate-pulse",
  completed: "bg-status-completed",
  failed: "bg-status-failed",
};

function compactPreview(
  viewType: string,
  preview: Record<string, unknown>
): string {
  if (viewType === "table") {
    const shape = preview.shape as { rows: number; cols: number } | undefined;
    if (shape) return `${shape.rows.toLocaleString()} rows x ${shape.cols} cols`;
    const rowCount = preview.row_count as number | undefined;
    if (rowCount) return `${rowCount.toLocaleString()} rows`;
    const train = preview.train as { row_count: number } | undefined;
    const test = preview.test as { row_count: number } | undefined;
    if (train && test) return `Train: ${train.row_count} / Test: ${test.row_count}`;
    return "Table";
  }
  if (viewType === "metrics") {
    const metrics = preview.metrics as Record<string, number> | undefined;
    if (metrics) {
      const entries = Object.entries(metrics);
      if (entries.length > 0) {
        const [name, val] = entries[0];
        return `${name}: ${typeof val === "number" ? val.toFixed(4) : val}`;
      }
    }
    return "Metrics";
  }
  if (viewType === "model_summary") {
    const mt = preview.model_type as string | undefined;
    return mt || "Model";
  }
  return "";
}

export const CardNode = memo(function CardNode({
  id,
  data,
  selected,
}: NodeProps<Node<CardNodeData>>) {
  const { cardSchema, status, outputPreview } = data;
  const isExecuting = usePipelineStore((s) => s.isExecuting);
  const removeNode = usePipelineStore((s) => s.removeNode);

  const inputKeys = Object.keys(cardSchema.input_schema);
  const outputKeys = Object.keys(cardSchema.output_schema);
  const maxHandles = Math.max(inputKeys.length, outputKeys.length, 1);

  const isRunning = status === "running";
  const colors = CATEGORY_COLORS[cardSchema.category] || CATEGORY_COLORS.data;

  return (
    <div
      className={`group relative rounded-lg border shadow-sm transition-all bg-bg-secondary w-56
        ${colors.border} ${colors.bg}
        ${selected ? "ring-2 ring-accent shadow-lg !border-accent" : "hover:shadow-md"}`}
    >
      {/* Delete button */}
      <button
        onClick={(e) => {
          e.stopPropagation();
          if (confirm("Remove this card?")) {
            removeNode(id);
          }
        }}
        className="absolute -top-2 -right-2 z-20
          w-5 h-5 flex items-center justify-center
          bg-bg-secondary border border-border rounded-full
          text-text-secondary/50 hover:text-status-failed
          hover:bg-status-failed/10 hover:border-status-failed/40
          transition-all shadow-sm
          opacity-0 group-hover:opacity-100"
        title="Remove card"
      >
        <svg className="w-2.5 h-2.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
        </svg>
      </button>

      {/* Card body */}
      <div className="px-3 py-2.5 flex flex-col gap-1">
        {/* Header row: title + status */}
        <div className="flex items-center justify-between">
          <h3 className="text-xs font-semibold text-foreground leading-tight truncate">
            {cardSchema.display_name}
          </h3>
          <div className="flex items-center gap-2.5 shrink-0 ml-2">
            {isRunning && (
              <span className="w-2 h-2 rounded-full bg-status-running animate-pulse" />
            )}
            {!isExecuting && !isRunning && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  runPipeline(id);
                }}
                className="text-text-secondary/40 hover:text-accent transition-colors p-0.5"
                title="Run from this node"
              >
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M8 5v14l11-7z" />
                </svg>
              </button>
            )}
            {!isRunning && (
              <span className={`w-2 h-2 rounded-full ${STATUS_DOT[status]}`} />
            )}
          </div>
        </div>

        {/* I/O ports section */}
        <div className="relative" style={{ minHeight: `${maxHandles * 22 + 8}px` }}>
          {/* Input handles */}
          {inputKeys.map((key, i) => {
            const top = i * 22 + 11;
            return (
              <div
                key={`in-${key}`}
                className="absolute left-0 flex items-center"
                style={{ top: `${top}px`, transform: "translateY(-50%)" }}
              >
                <Handle
                  type="target"
                  position={Position.Left}
                  id={key}
                  className="!w-2.5 !h-2.5 !bg-accent !border-2 !border-bg-secondary !shadow-sm hover:!bg-accent/80 transition-colors !rounded-full"
                  style={{ left: "-13px" }}
                />
                <span className="text-[9px] text-text-secondary/70 font-medium ml-0.5">
                  {key}
                </span>
              </div>
            );
          })}

          {/* Output handles */}
          {outputKeys.map((key, i) => {
            const top = i * 22 + 11;
            return (
              <div
                key={`out-${key}`}
                className="absolute right-0 flex items-center"
                style={{ top: `${top}px`, transform: "translateY(-50%)" }}
              >
                <span className="text-[9px] text-text-secondary/70 font-medium mr-0.5">
                  {key}
                </span>
                <Handle
                  type="source"
                  position={Position.Right}
                  id={key}
                  className="!w-2.5 !h-2.5 !bg-accent !border-2 !border-bg-secondary !shadow-sm hover:!bg-accent/80 transition-colors !rounded-full"
                  style={{ right: "-13px" }}
                />
              </div>
            );
          })}
        </div>

        {/* Compact output preview */}
        {status === "completed" && outputPreview && (
          <div className="text-[10px] text-status-completed/90 font-medium truncate text-center">
            {compactPreview(cardSchema.output_view_type, outputPreview.preview)}
          </div>
        )}

        {/* Error preview */}
        {status === "failed" && data.error && (
          <div className="text-[10px] text-status-failed truncate text-center">
            {data.error.substring(0, 60)}
          </div>
        )}

      </div>
    </div>
  );
});
