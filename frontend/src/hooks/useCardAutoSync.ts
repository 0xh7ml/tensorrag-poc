import { useEffect, useRef } from "react";
import { uploadCustomCard } from "@/lib/api";
import { useEditorStore } from "@/store/editorStore";

/**
 * Debounced auto-sync: uploads the active card file 1 second after edits stop.
 */
export function useCardAutoSync() {
  const cardFiles = useEditorStore((s) => s.cardFiles);
  const activeFilePath = useEditorStore((s) => s.activeFilePath);
  const markFileSaved = useEditorStore((s) => s.markFileSaved);
  const setIsSyncing = useEditorStore((s) => s.setIsSyncing);

  const timeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Find the active file's content fingerprint
  const activeFile = cardFiles.find((f) => f.path === activeFilePath);
  const content = activeFile?.content;
  const isDirty = activeFile?.isDirty;
  const fileName = activeFile?.name;

  useEffect(() => {
    if (!isDirty || !activeFilePath || !content || !fileName) return;

    // Clear previous timeout
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }

    timeoutRef.current = setTimeout(async () => {
      try {
        setIsSyncing(true);
        await uploadCustomCard(fileName, content);
        markFileSaved(activeFilePath);
      } catch {
        // Sync failure is non-critical — file stays dirty for retry
      } finally {
        setIsSyncing(false);
      }
    }, 1000);

    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, [content, isDirty, activeFilePath, fileName, markFileSaved, setIsSyncing]);
}
