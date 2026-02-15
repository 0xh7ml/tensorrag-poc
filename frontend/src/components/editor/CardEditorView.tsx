"use client";

import { useEffect, useRef, useCallback } from "react";
import Editor from "@monaco-editor/react";
import { useTheme } from "next-themes";
import { useEditorStore } from "@/store/editorStore";
import { listCustomCards } from "@/lib/api";
import { setupBaseCardIntelliSense } from "@/lib/services/monacoCardIntelliSense";
import { useCardAutoSync } from "@/hooks/useCardAutoSync";
import { CardFileTree } from "./CardFileTree";
import { EditorToolbar } from "./EditorToolbar";
import { ValidationPanel } from "./ValidationPanel";
import type { CardFile } from "@/store/editorStore";

export function CardEditorView() {
  const { theme } = useTheme();
  const cardFiles = useEditorStore((s) => s.cardFiles);
  const activeFilePath = useEditorStore((s) => s.activeFilePath);
  const updateFileContent = useEditorStore((s) => s.updateFileContent);
  const setCardFiles = useEditorStore((s) => s.setCardFiles);
  const setActiveFilePath = useEditorStore((s) => s.setActiveFilePath);
  const setIsLoadingFiles = useEditorStore((s) => s.setIsLoadingFiles);
  const isLoadingFiles = useEditorStore((s) => s.isLoadingFiles);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const editorRef = useRef<any>(null);

  const isDark =
    theme === "dark" ||
    (theme === "system" &&
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-color-scheme: dark)").matches);

  const activeFile = cardFiles.find((f) => f.path === activeFilePath);

  // Auto-sync hook
  useCardAutoSync();

  // Load custom card files on mount
  useEffect(() => {
    let cancelled = false;

    async function loadFiles() {
      setIsLoadingFiles(true);
      try {
        const customCards = await listCustomCards();
        if (cancelled) return;

        const files: CardFile[] = customCards.map((c) => ({
          name: c.filename,
          path: c.filename,
          content: c.source_code,
          language: "python" as const,
          isDirty: false,
        }));

        setCardFiles(files);
        if (files.length > 0 && !activeFilePath) {
          setActiveFilePath(files[0].path);
        }
      } catch {
        // Backend not reachable — start with empty file list
      } finally {
        if (!cancelled) setIsLoadingFiles(false);
      }
    }

    loadFiles();
    return () => {
      cancelled = true;
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const handleEditorDidMount = useCallback((editor: any, monaco: any) => {
    editorRef.current = editor;

    // Define custom theme
    monaco.editor.defineTheme("tensorrag-dark", {
      base: "vs-dark",
      inherit: true,
      rules: [
        { token: "", foreground: "d4d4d4" },
        { token: "comment", foreground: "6A9955" },
        { token: "keyword", foreground: "569CD6" },
        { token: "string", foreground: "CE9178" },
        { token: "number", foreground: "B5CEA8" },
        { token: "type", foreground: "4EC9B0" },
        { token: "function", foreground: "DCDCAA" },
      ],
      colors: {
        "editor.background": "#0d1117",
        "editor.foreground": "#d4d4d4",
        "editor.lineHighlightBackground": "#161b22",
        "editor.selectionBackground": "#264f78",
        "editorCursor.foreground": "#ffffff",
        "editorLineNumber.foreground": "#6e7681",
      },
    });

    monaco.editor.defineTheme("tensorrag-light", {
      base: "vs",
      inherit: true,
      rules: [],
      colors: {
        "editor.background": "#f8fafc",
        "editor.foreground": "#1e293b",
        "editor.lineHighlightBackground": "#f1f5f9",
        "editorLineNumber.foreground": "#94a3b8",
      },
    });

    // Set up BaseCard IntelliSense
    setupBaseCardIntelliSense(monaco);

    // Apply theme
    monaco.editor.setTheme(isDark ? "tensorrag-dark" : "tensorrag-light");
  }, [isDark]);

  // Update theme when it changes
  useEffect(() => {
    if (editorRef.current) {
      const monaco = (window as any).monaco; // eslint-disable-line @typescript-eslint/no-explicit-any
      if (monaco) {
        monaco.editor.setTheme(isDark ? "tensorrag-dark" : "tensorrag-light");
      }
    }
  }, [isDark]);

  function handleEditorChange(value: string | undefined) {
    if (value !== undefined && activeFilePath) {
      updateFileContent(activeFilePath, value);
    }
  }

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      <EditorToolbar />

      <div className="flex-1 flex overflow-hidden relative">
        <CardFileTree />

        {/* Editor panel */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {activeFile ? (
            <Editor
              height="100%"
              language="python"
              theme={isDark ? "tensorrag-dark" : "tensorrag-light"}
              value={activeFile.content}
              onChange={handleEditorChange}
              onMount={handleEditorDidMount}
              options={{
                fontSize: 14,
                fontFamily:
                  '"JetBrains Mono", "Fira Code", "SF Mono", Monaco, Consolas, monospace',
                fontLigatures: true,
                minimap: { enabled: false },
                lineNumbers: "on",
                wordWrap: "on",
                tabSize: 4,
                insertSpaces: true,
                scrollBeyondLastLine: false,
                automaticLayout: true,
                cursorBlinking: "smooth",
                bracketPairColorization: { enabled: true },
                suggestOnTriggerCharacters: true,
                quickSuggestions: {
                  other: true,
                  comments: false,
                  strings: false,
                },
                parameterHints: { enabled: true },
                hover: { enabled: true, delay: 300 },
                snippetSuggestions: "top",
              }}
              loading={
                <div className="flex-1 flex items-center justify-center">
                  <div className="w-6 h-6 border-2 border-accent border-t-transparent rounded-full animate-spin" />
                </div>
              }
            />
          ) : (
            <div className="flex-1 flex items-center justify-center">
              <div className="text-center text-text-secondary">
                <svg
                  className="w-12 h-12 mx-auto mb-3 text-text-secondary/40"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={1.5}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M17.25 6.75L22.5 12l-5.25 5.25m-10.5 0L1.5 12l5.25-5.25m7.5-3l-4.5 16.5"
                  />
                </svg>
                <p className="text-sm">
                  {isLoadingFiles
                    ? "Loading card files..."
                    : "Create or select a card file to start editing"}
                </p>
              </div>
            </div>
          )}
        </div>

        <ValidationPanel />
      </div>
    </div>
  );
}
