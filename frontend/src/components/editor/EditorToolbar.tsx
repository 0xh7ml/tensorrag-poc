"use client";

import { useEditorStore } from "@/store/editorStore";
import { validateCardCode } from "@/lib/api";
import { Play, Cloud, CloudOff, Loader2 } from "lucide-react";

export function EditorToolbar() {
  const activeFilePath = useEditorStore((s) => s.activeFilePath);
  const cardFiles = useEditorStore((s) => s.cardFiles);
  const isValidating = useEditorStore((s) => s.isValidating);
  const setIsValidating = useEditorStore((s) => s.setIsValidating);
  const setValidationResult = useEditorStore((s) => s.setValidationResult);
  const isSyncing = useEditorStore((s) => s.isSyncing);

  const activeFile = cardFiles.find((f) => f.path === activeFilePath);

  async function handleValidate() {
    if (!activeFile) return;

    setIsValidating(true);
    setValidationResult(null);

    try {
      const result = await validateCardCode(activeFile.content);
      setValidationResult(result);
    } catch (err) {
      setValidationResult({
        success: false,
        errors: [
          {
            line: null,
            message: err instanceof Error ? err.message : "Validation failed",
            severity: "error",
          },
        ],
        extracted_schema: null,
      });
    } finally {
      setIsValidating(false);
    }
  }

  return (
    <div className="h-10 flex items-center justify-between px-4 border-b border-border bg-bg-secondary shrink-0">
      {/* Left: file info */}
      <div className="flex items-center gap-2 text-xs">
        {activeFile ? (
          <>
            <span className="font-medium text-foreground">
              {activeFile.name}
            </span>
            {activeFile.isDirty && (
              <span className="w-1.5 h-1.5 rounded-full bg-accent" />
            )}
            <span className="text-text-secondary">Python</span>
          </>
        ) : (
          <span className="text-text-secondary">No file selected</span>
        )}
      </div>

      {/* Right: actions */}
      <div className="flex items-center gap-2">
        {/* Sync indicator */}
        <div className="flex items-center gap-1 text-text-secondary">
          {isSyncing ? (
            <Loader2 size={13} className="animate-spin" />
          ) : (
            <Cloud size={13} className="text-status-completed" />
          )}
          <span className="text-[10px]">
            {isSyncing ? "Syncing..." : "Saved"}
          </span>
        </div>

        {/* Validate button */}
        <button
          onClick={handleValidate}
          disabled={!activeFile || isValidating}
          className="flex items-center gap-1.5 px-3 py-1 text-xs font-medium rounded bg-accent text-white hover:bg-accent/90 disabled:opacity-50 transition-colors"
        >
          {isValidating ? (
            <Loader2 size={13} className="animate-spin" />
          ) : (
            <Play size={13} />
          )}
          {isValidating ? "Validating..." : "Validate"}
        </button>
      </div>
    </div>
  );
}
