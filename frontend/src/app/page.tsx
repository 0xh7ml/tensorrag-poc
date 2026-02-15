"use client";

import { useEffect } from "react";
import { ReactFlowProvider } from "@xyflow/react";
import { usePipelineStore } from "@/store/pipelineStore";
import { fetchCards } from "@/lib/api";
import { Header } from "@/components/header/Header";
import { CardPalette } from "@/components/sidebar/CardPalette";
import { PipelineCanvas } from "@/components/canvas/PipelineCanvas";
import { ConsolePanel } from "@/components/console/ConsolePanel";

function PipelineApp() {
  const setCardSchemas = usePipelineStore((s) => s.setCardSchemas);
  const validationErrors = usePipelineStore((s) => s.validationErrors);
  const setValidationErrors = usePipelineStore((s) => s.setValidationErrors);

  useEffect(() => {
    fetchCards().then(setCardSchemas).catch(console.error);
  }, [setCardSchemas]);

  return (
    <div className="h-screen flex flex-col">
      <Header />

      {validationErrors.length > 0 && (
        <div className="bg-status-failed/10 border-b border-status-failed/30 px-4 py-2 flex items-start gap-2">
          <div className="flex-1">
            {validationErrors.map((err, i) => (
              <p key={i} className="text-xs text-status-failed">
                {err}
              </p>
            ))}
          </div>
          <button
            onClick={() => setValidationErrors([])}
            className="text-xs text-status-failed hover:text-status-failed/70"
          >
            Dismiss
          </button>
        </div>
      )}

      <div className="flex-1 flex overflow-hidden relative">
        <CardPalette />
        <div className="flex-1 flex flex-col overflow-hidden relative">
          <PipelineCanvas />
          <ConsolePanel />
        </div>
      </div>
    </div>
  );
}

export default function Home() {
  return (
    <ReactFlowProvider>
      <PipelineApp />
    </ReactFlowProvider>
  );
}
