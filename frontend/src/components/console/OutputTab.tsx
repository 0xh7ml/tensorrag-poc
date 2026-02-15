"use client";

import { useTheme } from "next-themes";
import { usePipelineStore } from "@/store/pipelineStore";
import { OutputViewer } from "@/components/output/OutputViewer";

export function OutputTab() {
  const { theme } = useTheme();
  const selectedNodeIdForOutput = usePipelineStore((s) => s.selectedNodeIdForOutput);
  const nodes = usePipelineStore((s) => s.nodes);
  const pipelineId = usePipelineStore((s) => s.pipelineId);
  
  const isDark = theme === "dark" || (theme === "system" && typeof window !== "undefined" && window.matchMedia("(prefers-color-scheme: dark)").matches);

  const selectedNode = selectedNodeIdForOutput
    ? nodes.find((n) => n.id === selectedNodeIdForOutput)
    : null;

  if (!selectedNodeIdForOutput || !selectedNode) {
    return (
      <div className={`flex-1 flex items-center justify-center ${isDark ? "text-gray-400" : "text-gray-500"}`}>
        <div className="text-center">
          <svg
            className={`w-12 h-12 mx-auto mb-3 ${isDark ? "text-gray-600" : "text-gray-400"}`}
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={1.5}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
            />
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
            />
          </svg>
          <p className={`text-sm ${isDark ? "text-gray-300" : "text-gray-700"}`}>No output selected</p>
          <p className={`text-xs mt-1 ${isDark ? "text-gray-500" : "text-gray-500"}`}>
            Click a card on the canvas to view its output here
          </p>
        </div>
      </div>
    );
  }

  const { cardSchema, status, error, outputPreview } = selectedNode.data;

  return (
    <div className={`flex-1 overflow-y-auto p-4 pb-8 ${isDark ? "text-gray-200" : "text-gray-700"}`}>
      {/* Header info */}
      <div className="flex items-center gap-2 mb-4">
        <h3 className={`text-sm font-semibold ${isDark ? "text-white" : "text-gray-900"}`}>{cardSchema.display_name}</h3>
        <span
          className={`text-[10px] px-1.5 py-0.5 rounded font-medium
            ${status === "running" ? isDark ? "bg-yellow-500/20 text-yellow-400" : "bg-yellow-100 text-yellow-700" : ""}
            ${status === "completed" ? isDark ? "bg-green-500/20 text-green-400" : "bg-green-100 text-green-700" : ""}
            ${status === "failed" ? isDark ? "bg-red-500/20 text-red-400" : "bg-red-100 text-red-700" : ""}
            ${status === "pending" ? isDark ? "bg-gray-500/20 text-gray-400" : "bg-gray-100 text-gray-700" : ""}
          `}
        >
          {status}
        </span>
        {cardSchema.description && (
          <span className={`text-xs ${isDark ? "text-gray-500" : "text-gray-400"}`}>&mdash; {cardSchema.description}</span>
        )}
      </div>

      {/* Content */}
      {status === "running" && (
        <div className="flex flex-col items-center justify-center py-12 gap-3">
          <div className={`w-8 h-8 border-2 border-blue-400 border-t-transparent rounded-full animate-spin`} />
          <p className={`text-sm ${isDark ? "text-gray-300" : "text-gray-700"}`}>Executing...</p>
        </div>
      )}

      {status === "pending" && (
        <div className="flex flex-col items-center justify-center py-12">
          <p className={`text-sm ${isDark ? "text-gray-300" : "text-gray-700"}`}>Waiting to execute</p>
        </div>
      )}

      {status === "failed" && error && (
        <div className={`p-3 rounded-md ${isDark ? "bg-red-500/10 border-red-500/30" : "bg-red-50 border-red-200"} border`}>
          <h3 className={`text-xs font-semibold mb-1 ${isDark ? "text-red-400" : "text-red-700"}`}>Error</h3>
          <p className={`text-xs whitespace-pre-wrap ${isDark ? "text-red-300" : "text-red-600"}`}>{error}</p>
        </div>
      )}

      {status === "completed" && outputPreview && (
        <div className={isDark ? "bg-[#0d1117]" : "bg-white"}>
          <OutputViewer
            outputType={outputPreview.output_type}
            preview={outputPreview.preview}
            pipelineId={pipelineId || ""}
            nodeId={selectedNodeIdForOutput}
          />
        </div>
      )}

      {status === "completed" && !outputPreview && (
        <div className="flex flex-col items-center justify-center py-12">
          <p className={`text-sm ${isDark ? "text-gray-300" : "text-gray-700"}`}>No output preview available</p>
        </div>
      )}
    </div>
  );
}
