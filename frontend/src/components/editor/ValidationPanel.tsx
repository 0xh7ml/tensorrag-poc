"use client";

import { useEditorStore } from "@/store/editorStore";
import { usePipelineStore } from "@/store/pipelineStore";
import { uploadCustomCard, fetchCards } from "@/lib/api";
import {
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Loader2,
  Upload,
} from "lucide-react";
import { useState } from "react";

export function ValidationPanel() {
  const activeFilePath = useEditorStore((s) => s.activeFilePath);
  const cardFiles = useEditorStore((s) => s.cardFiles);
  const isValidating = useEditorStore((s) => s.isValidating);
  const validationResult = useEditorStore((s) => s.validationResult);
  const setCardSchemas = usePipelineStore((s) => s.setCardSchemas);
  const setActiveView = usePipelineStore((s) => s.setActiveView);

  const [isPublishing, setIsPublishing] = useState(false);
  const [publishStatus, setPublishStatus] = useState<string | null>(null);

  const activeFile = cardFiles.find((f) => f.path === activeFilePath);

  async function handlePublish() {
    if (!activeFile || !validationResult?.success) return;

    setIsPublishing(true);
    setPublishStatus(null);

    try {
      await uploadCustomCard(activeFile.name, activeFile.content);

      // Refresh card schemas so the palette picks up the new card
      const schemas = await fetchCards();
      setCardSchemas(schemas);

      setPublishStatus("success");

      // Auto-switch to board after a short delay
      setTimeout(() => {
        setActiveView("board");
      }, 1200);
    } catch (err) {
      setPublishStatus(
        err instanceof Error ? err.message : "Publish failed"
      );
    } finally {
      setIsPublishing(false);
    }
  }

  return (
    <aside className="w-72 border-l border-border bg-bg-secondary shrink-0 flex flex-col overflow-hidden">
      {/* Header */}
      <div className="px-3 py-2.5 border-b border-border">
        <h2 className="text-[11px] font-semibold uppercase tracking-wider text-text-secondary">
          Validation
        </h2>
      </div>

      <div className="flex-1 overflow-y-auto p-3">
        {/* No file selected */}
        {!activeFile && (
          <div className="text-xs text-text-secondary text-center py-8">
            Select a card file to validate.
          </div>
        )}

        {/* Validating spinner */}
        {activeFile && isValidating && (
          <div className="flex flex-col items-center gap-2 py-8">
            <Loader2 size={24} className="animate-spin text-accent" />
            <span className="text-xs text-text-secondary">Validating...</span>
          </div>
        )}

        {/* Idle state */}
        {activeFile && !isValidating && !validationResult && (
          <div className="text-xs text-text-secondary text-center py-8">
            Click <strong>Validate</strong> to check your card code.
          </div>
        )}

        {/* Validation errors */}
        {activeFile && !isValidating && validationResult && !validationResult.success && (
          <div>
            <div className="flex items-center gap-1.5 mb-3">
              <XCircle size={16} className="text-status-failed" />
              <span className="text-xs font-semibold text-status-failed">
                {validationResult.errors.length} error
                {validationResult.errors.length > 1 ? "s" : ""} found
              </span>
            </div>

            <div className="flex flex-col gap-2">
              {validationResult.errors.map((err, i) => (
                <div
                  key={i}
                  className="flex items-start gap-2 p-2 rounded bg-status-failed/10 border border-status-failed/20"
                >
                  {err.severity === "warning" ? (
                    <AlertTriangle
                      size={13}
                      className="text-status-running shrink-0 mt-0.5"
                    />
                  ) : (
                    <XCircle
                      size={13}
                      className="text-status-failed shrink-0 mt-0.5"
                    />
                  )}
                  <div className="flex-1">
                    <p className="text-[11px] text-foreground">{err.message}</p>
                    {err.line && (
                      <p className="text-[10px] text-text-secondary mt-0.5">
                        Line {err.line}
                      </p>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Validation success */}
        {activeFile && !isValidating && validationResult?.success && validationResult.extracted_schema && (
          <div>
            <div className="flex items-center gap-1.5 mb-3">
              <CheckCircle2 size={16} className="text-status-completed" />
              <span className="text-xs font-semibold text-status-completed">
                Valid card
              </span>
            </div>

            {/* Extracted schema preview */}
            <div className="flex flex-col gap-2 text-xs">
              <SchemaField label="Type" value={validationResult.extracted_schema.card_type} />
              <SchemaField label="Name" value={validationResult.extracted_schema.display_name} />
              <SchemaField label="Category" value={validationResult.extracted_schema.category} />
              <SchemaField label="Mode" value={validationResult.extracted_schema.execution_mode} />
              <SchemaField label="View" value={validationResult.extracted_schema.output_view_type} />

              {Object.keys(validationResult.extracted_schema.input_schema || {}).length > 0 && (
                <div>
                  <span className="text-text-secondary font-medium">Inputs:</span>
                  <div className="ml-2 mt-0.5">
                    {Object.entries(validationResult.extracted_schema.input_schema).map(
                      ([k, v]) => (
                        <div key={k} className="text-[11px] text-foreground">
                          {k}: <span className="text-accent">{String(v)}</span>
                        </div>
                      )
                    )}
                  </div>
                </div>
              )}

              {Object.keys(validationResult.extracted_schema.output_schema || {}).length > 0 && (
                <div>
                  <span className="text-text-secondary font-medium">Outputs:</span>
                  <div className="ml-2 mt-0.5">
                    {Object.entries(validationResult.extracted_schema.output_schema).map(
                      ([k, v]) => (
                        <div key={k} className="text-[11px] text-foreground">
                          {k}: <span className="text-accent">{String(v)}</span>
                        </div>
                      )
                    )}
                  </div>
                </div>
              )}
            </div>

            {/* Publish button */}
            <button
              onClick={handlePublish}
              disabled={isPublishing}
              className="mt-4 w-full flex items-center justify-center gap-1.5 px-3 py-2 text-xs font-medium rounded bg-status-completed text-white hover:bg-status-completed/90 disabled:opacity-50 transition-colors"
            >
              {isPublishing ? (
                <Loader2 size={13} className="animate-spin" />
              ) : (
                <Upload size={13} />
              )}
              {isPublishing ? "Publishing..." : "Publish to Board"}
            </button>

            {publishStatus === "success" && (
              <p className="text-[10px] text-status-completed text-center mt-2">
                Published! Switching to Board...
              </p>
            )}
            {publishStatus && publishStatus !== "success" && (
              <p className="text-[10px] text-status-failed text-center mt-2">
                {publishStatus}
              </p>
            )}
          </div>
        )}
      </div>
    </aside>
  );
}

function SchemaField({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-text-secondary">{label}</span>
      <span className="text-foreground font-medium">{value}</span>
    </div>
  );
}
