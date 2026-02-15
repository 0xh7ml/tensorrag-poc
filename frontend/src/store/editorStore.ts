import { create } from "zustand";
import type { CardValidationResult } from "@/lib/types";

export interface CardFile {
  name: string;
  path: string;
  content: string;
  language: "python";
  isDirty: boolean;
}

interface EditorState {
  // File management
  cardFiles: CardFile[];
  activeFilePath: string | null;
  setActiveFilePath: (path: string | null) => void;

  // File CRUD
  addCardFile: (file: CardFile) => void;
  updateFileContent: (path: string, content: string) => void;
  removeCardFile: (path: string) => void;
  renameCardFile: (oldPath: string, newPath: string) => void;
  setCardFiles: (files: CardFile[]) => void;
  markFileSaved: (path: string) => void;

  // Validation
  isValidating: boolean;
  validationResult: CardValidationResult | null;
  setIsValidating: (v: boolean) => void;
  setValidationResult: (r: CardValidationResult | null) => void;

  // Sync
  isSyncing: boolean;
  isLoadingFiles: boolean;
  setIsSyncing: (v: boolean) => void;
  setIsLoadingFiles: (v: boolean) => void;
}

export const useEditorStore = create<EditorState>()((set, get) => ({
  cardFiles: [],
  activeFilePath: null,
  setActiveFilePath: (path) => set({ activeFilePath: path }),

  addCardFile: (file) => {
    set({
      cardFiles: [...get().cardFiles, file],
      activeFilePath: file.path,
    });
  },

  updateFileContent: (path, content) => {
    set({
      cardFiles: get().cardFiles.map((f) =>
        f.path === path ? { ...f, content, isDirty: true } : f
      ),
    });
  },

  removeCardFile: (path) => {
    const files = get().cardFiles.filter((f) => f.path !== path);
    const active = get().activeFilePath;
    set({
      cardFiles: files,
      activeFilePath:
        active === path ? (files.length > 0 ? files[0].path : null) : active,
    });
  },

  renameCardFile: (oldPath, newPath) => {
    const newName = newPath.split("/").pop() || newPath;
    set({
      cardFiles: get().cardFiles.map((f) =>
        f.path === oldPath ? { ...f, path: newPath, name: newName } : f
      ),
      activeFilePath: get().activeFilePath === oldPath ? newPath : get().activeFilePath,
    });
  },

  setCardFiles: (files) => set({ cardFiles: files }),

  markFileSaved: (path) => {
    set({
      cardFiles: get().cardFiles.map((f) =>
        f.path === path ? { ...f, isDirty: false } : f
      ),
    });
  },

  isValidating: false,
  validationResult: null,
  setIsValidating: (v) => set({ isValidating: v }),
  setValidationResult: (r) => set({ validationResult: r }),

  isSyncing: false,
  isLoadingFiles: false,
  setIsSyncing: (v) => set({ isSyncing: v }),
  setIsLoadingFiles: (v) => set({ isLoadingFiles: v }),
}));
