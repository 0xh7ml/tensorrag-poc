"use client";

import { useState } from "react";
import { useEditorStore, type CardFile } from "@/store/editorStore";
import { generateCardTemplate } from "@/lib/cardTemplate";
import { deleteCustomCard } from "@/lib/api";
import { Plus, FileCode2, Trash2, Pencil } from "lucide-react";

export function CardFileTree() {
  const cardFiles = useEditorStore((s) => s.cardFiles);
  const activeFilePath = useEditorStore((s) => s.activeFilePath);
  const setActiveFilePath = useEditorStore((s) => s.setActiveFilePath);
  const addCardFile = useEditorStore((s) => s.addCardFile);
  const removeCardFile = useEditorStore((s) => s.removeCardFile);
  const renameCardFile = useEditorStore((s) => s.renameCardFile);
  const isLoadingFiles = useEditorStore((s) => s.isLoadingFiles);

  const [showNewDialog, setShowNewDialog] = useState(false);
  const [newCardName, setNewCardName] = useState("");
  const [hoveredFile, setHoveredFile] = useState<string | null>(null);

  const [showRenameDialog, setShowRenameDialog] = useState(false);
  const [renamePath, setRenamePath] = useState("");
  const [renameValue, setRenameValue] = useState("");

  function handleCreateCard() {
    const name = newCardName.trim();
    if (!name) return;

    // Normalize: ensure .py extension, use snake_case for card_type
    const fileName = name.endsWith(".py") ? name : `${name}.py`;
    const cardType = fileName.replace(/\.py$/, "");
    const displayName = cardType
      .split("_")
      .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
      .join(" ");

    // Check for duplicates
    if (cardFiles.some((f) => f.name === fileName)) {
      alert(`A card file named "${fileName}" already exists.`);
      return;
    }

    const content = generateCardTemplate(cardType, displayName);
    const file: CardFile = {
      name: fileName,
      path: fileName,
      content,
      language: "python",
      isDirty: true,
    };

    addCardFile(file);
    setNewCardName("");
    setShowNewDialog(false);
  }

  async function handleDeleteFile(path: string) {
    const file = cardFiles.find((f) => f.path === path);
    if (!file) return;

    if (!confirm(`Delete "${file.name}"?`)) return;

    // Try to extract card_type from the filename
    const cardType = file.name.replace(/\.py$/, "");
    try {
      await deleteCustomCard(cardType);
    } catch {
      // File might not be published yet — that's fine
    }

    removeCardFile(path);
  }

  function handleStartRename(path: string) {
    const file = cardFiles.find((f) => f.path === path);
    if (!file) return;
    setRenamePath(path);
    setRenameValue(file.name);
    setShowRenameDialog(true);
  }

  function handleConfirmRename() {
    const newName = renameValue.trim();
    if (!newName || newName === renamePath) {
      setShowRenameDialog(false);
      return;
    }

    const fileName = newName.endsWith(".py") ? newName : `${newName}.py`;

    if (cardFiles.some((f) => f.path === fileName && f.path !== renamePath)) {
      alert(`A file named "${fileName}" already exists.`);
      return;
    }

    renameCardFile(renamePath, fileName);
    setShowRenameDialog(false);
  }

  return (
    <aside className="w-52 border-r border-border bg-bg-secondary shrink-0 flex flex-col overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2.5 border-b border-border">
        <h2 className="text-[11px] font-semibold uppercase tracking-wider text-text-secondary">
          Card Files
        </h2>
        <button
          onClick={() => setShowNewDialog(true)}
          className="p-1 rounded hover:bg-border/40 text-text-secondary hover:text-foreground transition-colors"
          title="New card file"
        >
          <Plus size={14} />
        </button>
      </div>

      {/* File list */}
      <div className="flex-1 overflow-y-auto py-1">
        {isLoadingFiles ? (
          <div className="flex items-center justify-center py-8">
            <div className="w-5 h-5 border-2 border-accent border-t-transparent rounded-full animate-spin" />
          </div>
        ) : cardFiles.length === 0 ? (
          <div className="px-3 py-8 text-center text-xs text-text-secondary">
            No card files yet.
            <br />
            Click + to create one.
          </div>
        ) : (
          cardFiles.map((file) => (
            <div
              key={file.path}
              onClick={() => setActiveFilePath(file.path)}
              onMouseEnter={() => setHoveredFile(file.path)}
              onMouseLeave={() => setHoveredFile(null)}
              className={`flex items-center gap-2 px-3 py-1.5 cursor-pointer transition-colors group ${
                file.path === activeFilePath
                  ? "bg-accent/15 text-foreground"
                  : "text-text-secondary hover:bg-border/20 hover:text-foreground"
              }`}
            >
              <FileCode2 size={14} className="text-yellow-500 shrink-0" />
              <span className="flex-1 text-xs truncate">{file.name}</span>
              {file.isDirty && (
                <span className="w-1.5 h-1.5 rounded-full bg-accent shrink-0" />
              )}
              {hoveredFile === file.path && (
                <div className="flex items-center gap-0.5 shrink-0">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleStartRename(file.path);
                    }}
                    className="p-0.5 rounded hover:bg-border/40 text-text-secondary/60 hover:text-foreground"
                    title="Rename"
                  >
                    <Pencil size={11} />
                  </button>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDeleteFile(file.path);
                    }}
                    className="p-0.5 rounded hover:bg-status-failed/10 text-text-secondary/60 hover:text-status-failed"
                    title="Delete"
                  >
                    <Trash2 size={11} />
                  </button>
                </div>
              )}
            </div>
          ))
        )}
      </div>

      {/* New card dialog */}
      {showNewDialog && (
        <div className="absolute inset-0 z-50 flex items-center justify-center bg-background/80">
          <div className="bg-bg-secondary border border-border rounded-lg shadow-lg p-4 w-72">
            <h3 className="text-sm font-semibold text-foreground mb-3">
              New Card File
            </h3>
            <input
              type="text"
              value={newCardName}
              onChange={(e) => setNewCardName(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleCreateCard()}
              placeholder="e.g. feature_scaler"
              className="w-full px-3 py-1.5 text-sm rounded border border-border bg-background text-foreground focus:outline-none focus:ring-1 focus:ring-accent"
              autoFocus
            />
            <p className="text-[10px] text-text-secondary mt-1.5">
              Use snake_case. The .py extension will be added automatically.
            </p>
            <div className="flex justify-end gap-2 mt-3">
              <button
                onClick={() => {
                  setShowNewDialog(false);
                  setNewCardName("");
                }}
                className="px-3 py-1 text-xs rounded border border-border text-text-secondary hover:text-foreground"
              >
                Cancel
              </button>
              <button
                onClick={handleCreateCard}
                disabled={!newCardName.trim()}
                className="px-3 py-1 text-xs rounded bg-accent text-white hover:bg-accent/90 disabled:opacity-50"
              >
                Create
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Rename dialog */}
      {showRenameDialog && (
        <div className="absolute inset-0 z-50 flex items-center justify-center bg-background/80">
          <div className="bg-bg-secondary border border-border rounded-lg shadow-lg p-4 w-72">
            <h3 className="text-sm font-semibold text-foreground mb-3">
              Rename File
            </h3>
            <input
              type="text"
              value={renameValue}
              onChange={(e) => setRenameValue(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleConfirmRename()}
              className="w-full px-3 py-1.5 text-sm rounded border border-border bg-background text-foreground focus:outline-none focus:ring-1 focus:ring-accent"
              autoFocus
            />
            <div className="flex justify-end gap-2 mt-3">
              <button
                onClick={() => setShowRenameDialog(false)}
                className="px-3 py-1 text-xs rounded border border-border text-text-secondary hover:text-foreground"
              >
                Cancel
              </button>
              <button
                onClick={handleConfirmRename}
                className="px-3 py-1 text-xs rounded bg-accent text-white hover:bg-accent/90"
              >
                Rename
              </button>
            </div>
          </div>
        </div>
      )}
    </aside>
  );
}
