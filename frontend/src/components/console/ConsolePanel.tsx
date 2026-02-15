"use client";

import { useEffect, useRef, useState } from "react";
import { useTheme } from "next-themes";
import { usePipelineStore } from "@/store/pipelineStore";
import { OutputTab } from "./OutputTab";
import { SchemaForm } from "@/components/config/SchemaForm";

const MIN_HEIGHT = 100;
const MAX_HEIGHT = 600;

export function ConsolePanel() {
  const { theme } = useTheme();
  const consoleOpen = usePipelineStore((s) => s.consoleOpen);
  const setConsoleOpen = usePipelineStore((s) => s.setConsoleOpen);
  const consoleHeight = usePipelineStore((s) => s.consoleHeight);
  const setConsoleHeight = usePipelineStore((s) => s.setConsoleHeight);
  const consoleActiveTab = usePipelineStore((s) => s.consoleActiveTab);
  const setConsoleActiveTab = usePipelineStore((s) => s.setConsoleActiveTab);
  const logEntries = usePipelineStore((s) => s.logEntries);
  const isExecuting = usePipelineStore((s) => s.isExecuting);
  const selectedNodeId = usePipelineStore((s) => s.selectedNodeId);
  const nodes = usePipelineStore((s) => s.nodes);
  const updateNodeConfig = usePipelineStore((s) => s.updateNodeConfig);
  const selectedNode = nodes.find((n) => n.id === selectedNodeId);
  const logBottomRef = useRef<HTMLDivElement>(null);
  const [isResizing, setIsResizing] = useState(false);
  const resizeRef = useRef<HTMLDivElement>(null);
  
  const isDark = theme === "dark" || (theme === "system" && typeof window !== "undefined" && window.matchMedia("(prefers-color-scheme: dark)").matches);

  useEffect(() => {
    if (consoleOpen && logBottomRef.current) {
      logBottomRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [logEntries.length, consoleOpen]);

  useEffect(() => {
    if (!isResizing) return;

    const handleMouseMove = (e: MouseEvent) => {
      const newHeight = window.innerHeight - e.clientY;
      setConsoleHeight(newHeight);
    };

    const handleMouseUp = () => {
      setIsResizing(false);
    };

    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);

    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isResizing, setConsoleHeight]);

  const handleMinimize = () => {
    setConsoleHeight(MIN_HEIGHT);
  };

  const handleMaximize = () => {
    setConsoleHeight(MAX_HEIGHT);
  };

  if (!consoleOpen) return null;

  return (
    <div 
      className={`absolute bottom-0 left-0 right-0 border-t border-border/40 flex flex-col ${
        isDark ? "bg-[#0d1117]" : "bg-gray-100"
      }`} 
      style={{ height: `${consoleHeight}px` }}
    >
      {/* Resize handle */}
      <div
        ref={resizeRef}
        onMouseDown={(e) => {
          e.preventDefault();
          setIsResizing(true);
        }}
        className="h-1 cursor-ns-resize hover:bg-accent/30 transition-colors group"
        title="Drag to resize"
      >
        <div className="h-full w-full flex items-center justify-center">
          <div className="w-12 h-0.5 bg-border/40 group-hover:bg-accent/60 rounded-full transition-colors" />
        </div>
      </div>

      {/* Tabs */}
      <div className={`flex items-center justify-between border-b border-border/40 ${isDark ? "bg-[#161b22]" : "bg-gray-200"}`}>
        <div className="flex items-center">
          <button
            onClick={() => setConsoleActiveTab("console")}
            className={`px-4 py-2 text-sm font-medium transition-colors border-b-2 ${
              consoleActiveTab === "console"
                ? isDark 
                  ? "text-white border-accent"
                  : "text-gray-900 border-blue-500"
                : isDark
                  ? "text-gray-400 border-transparent hover:text-gray-200"
                  : "text-gray-600 border-transparent hover:text-gray-900"
            }`}
          >
            <span className="flex items-center gap-2">
              Console
              {isExecuting && consoleActiveTab === "console" && (
                <span className={`text-xs ${isDark ? "text-gray-400" : "text-gray-500"}`}>(Running...)</span>
              )}
            </span>
          </button>
          <button
            onClick={() => setConsoleActiveTab("output")}
            className={`px-4 py-2 text-sm font-medium transition-colors border-b-2 ${
              consoleActiveTab === "output"
                ? isDark
                  ? "text-white border-accent"
                  : "text-gray-900 border-blue-500"
                : isDark
                  ? "text-gray-400 border-transparent hover:text-gray-200"
                  : "text-gray-600 border-transparent hover:text-gray-900"
            }`}
          >
            Output
          </button>
          <button
            onClick={() => setConsoleActiveTab("config")}
            className={`px-4 py-2 text-sm font-medium transition-colors border-b-2 ${
              consoleActiveTab === "config"
                ? isDark
                  ? "text-white border-accent"
                  : "text-gray-900 border-blue-500"
                : isDark
                  ? "text-gray-400 border-transparent hover:text-gray-200"
                  : "text-gray-600 border-transparent hover:text-gray-900"
            }`}
          >
            <span className="flex items-center gap-2">
              Config
              {selectedNode && (
                <span className={`text-xs ${isDark ? "text-gray-400" : "text-gray-500"}`}>
                  ({selectedNode.data.cardSchema.display_name})
                </span>
              )}
            </span>
          </button>
        </div>
        <div className="flex items-center gap-2 pr-2">
          <button
            onClick={handleMinimize}
            className={`transition-colors p-1 rounded ${
              isDark 
                ? "text-gray-400 hover:text-white hover:bg-white/10"
                : "text-gray-600 hover:text-gray-900 hover:bg-gray-300"
            }`}
            title="Minimize"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M20 12H4" />
            </svg>
          </button>
          <button
            onClick={handleMaximize}
            className={`transition-colors p-1 rounded ${
              isDark 
                ? "text-gray-400 hover:text-white hover:bg-white/10"
                : "text-gray-600 hover:text-gray-900 hover:bg-gray-300"
            }`}
            title="Maximize"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
            </svg>
          </button>
          {consoleActiveTab === "console" && (
            <button
              onClick={() => usePipelineStore.getState().clearLogs()}
              className={`text-xs transition-colors px-2 py-1 rounded ${
                isDark 
                  ? "text-gray-400 hover:text-white hover:bg-white/10"
                  : "text-gray-600 hover:text-gray-900 hover:bg-gray-300"
              }`}
              title="Clear console"
            >
              Clear
            </button>
          )}
          <button
            onClick={() => setConsoleOpen(false)}
            className={`transition-colors p-1 rounded ${
              isDark 
                ? "text-gray-400 hover:text-white hover:bg-white/10"
                : "text-gray-600 hover:text-gray-900 hover:bg-gray-300"
            }`}
            title="Close console"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
      </div>

      {/* Content */}
      {consoleActiveTab === "console" && (
        <div className={`flex-1 overflow-y-auto px-4 py-2 pb-8 font-mono text-xs leading-relaxed ${
          isDark ? "text-gray-300" : "text-gray-700"
        }`}>
          {logEntries.length === 0 && !isExecuting && (
            <div className={`text-center py-8 ${isDark ? "text-gray-500" : "text-gray-500"}`}>
              No logs yet. Console output will appear here when you run a pipeline.
            </div>
          )}
          {logEntries.map((entry, i) => (
            <div
              key={i}
              className={
                entry.text.includes("FAILED") || entry.text.includes("ERROR")
                  ? "text-red-500"
                  : entry.text.startsWith("$")
                    ? "text-green-600"
                    : isDark
                      ? "text-gray-300"
                      : "text-gray-700"
              }
            >
              {entry.text}
            </div>
          ))}
          {isExecuting && (
            <div className={`flex items-center gap-2 mt-2 ${isDark ? "text-gray-400" : "text-gray-600"}`}>
              <div className={`w-2 h-2 border border-accent border-t-transparent rounded-full animate-spin`} />
              <span>executing...</span>
            </div>
          )}
          <div ref={logBottomRef} />
        </div>
      )}
      {consoleActiveTab === "output" && (
        <div className={`flex-1 overflow-y-auto ${isDark ? "bg-[#0d1117]" : "bg-gray-50"}`}>
          <OutputTab />
        </div>
      )}
      {consoleActiveTab === "config" && (
        <div className={`flex-1 overflow-y-auto ${isDark ? "bg-[#0d1117]" : "bg-gray-50"}`}>
          {selectedNode ? (
            <div className="p-4">
              <div className="flex items-center justify-between mb-1">
                <h2 className={`text-sm font-semibold ${isDark ? "text-white" : "text-gray-900"}`}>
                  {selectedNode.data.cardSchema.display_name}
                </h2>
                <span className={`text-[10px] px-1.5 py-0.5 rounded ${isDark ? "bg-gray-700 text-gray-300" : "bg-border text-text-secondary"}`}>
                  {selectedNode.data.cardSchema.category}
                </span>
              </div>
              <p className={`text-xs mb-4 ${isDark ? "text-gray-400" : "text-text-secondary"}`}>
                {selectedNode.data.cardSchema.description}
              </p>

              <h3 className={`text-xs font-semibold uppercase mb-2 ${isDark ? "text-gray-400" : "text-text-secondary"}`}>
                Configuration
              </h3>
              <SchemaForm
                schema={selectedNode.data.cardSchema.config_schema}
                values={selectedNode.data.config}
                onChange={(config) => updateNodeConfig(selectedNode.id, config)}
              />


            </div>
          ) : (
            <div className={`text-center py-8 ${isDark ? "text-gray-500" : "text-gray-500"}`}>
              No card selected. Click a card on the canvas to configure it.
            </div>
          )}
        </div>
      )}
    </div>
  );
}
