"use client";

import { ThemeToggle } from "@/components/ThemeToggle";
import { usePipelineStore } from "@/store/pipelineStore";

export function Header() {
  const activeView = usePipelineStore((s) => s.activeView);
  const setActiveView = usePipelineStore((s) => s.setActiveView);

  return (
    <header className="h-12 flex items-center justify-between px-4 border-b border-border bg-bg-secondary shrink-0">
      <div className="flex items-center gap-2">
        <span className="font-semibold text-sm tracking-tight">TensorRag</span>
      </div>
      <div className="flex items-center bg-background rounded-lg border border-border p-0.5">
        <button
          onClick={() => setActiveView("board")}
          className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${
            activeView === "board"
              ? "bg-accent text-white"
              : "text-text-secondary hover:text-foreground"
          }`}
        >
          Board
        </button>
        <button
          onClick={() => setActiveView("editor")}
          className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${
            activeView === "editor"
              ? "bg-accent text-white"
              : "text-text-secondary hover:text-foreground"
          }`}
        >
          Editor
        </button>
      </div>
      <div className="flex items-center gap-3">
        <ThemeToggle />
      </div>
    </header>
  );
}
