"use client";

import { useCallback, useMemo } from "react";
import {
  ReactFlow,
  MiniMap,
  Controls,
  Background,
  BackgroundVariant,
  useReactFlow,
  type Node,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { usePipelineStore } from "@/store/pipelineStore";
import { CardNode } from "./CardNode";

const nodeTypes = { cardNode: CardNode };

export function PipelineCanvas() {
  const nodes = usePipelineStore((s) => s.nodes);
  const edges = usePipelineStore((s) => s.edges);
  const onNodesChange = usePipelineStore((s) => s.onNodesChange);
  const onEdgesChange = usePipelineStore((s) => s.onEdgesChange);
  const onConnect = usePipelineStore((s) => s.onConnect);
  const addNode = usePipelineStore((s) => s.addNode);
  const cardSchemas = usePipelineStore((s) => s.cardSchemas);
  const selectNode = usePipelineStore((s) => s.selectNode);
  const setConsoleOpen = usePipelineStore((s) => s.setConsoleOpen);
  const setConsoleActiveTab = usePipelineStore((s) => s.setConsoleActiveTab);
  const consoleOpen = usePipelineStore((s) => s.consoleOpen);
  const consoleHeight = usePipelineStore((s) => s.consoleHeight);

  const { screenToFlowPosition } = useReactFlow();

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      const cardType = event.dataTransfer.getData("application/tensorrag-card");
      if (!cardType) return;

      const schema = cardSchemas.find((s) => s.card_type === cardType);
      if (!schema) return;

      const position = screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });
      addNode(schema, position);
    },
    [cardSchemas, addNode, screenToFlowPosition]
  );

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
  }, []);

  const onNodeClick = useCallback(
    (_: React.MouseEvent, node: Node) => {
      selectNode(node.id);
      setConsoleActiveTab("config");
      setConsoleOpen(true);
    },
    [selectNode, setConsoleActiveTab, setConsoleOpen]
  );

  const onPaneClick = useCallback(() => {
    selectNode(null);
  }, [selectNode]);

  const stableNodeTypes = useMemo(() => nodeTypes, []);

  return (
    <div className="flex-1 relative" style={{ paddingBottom: consoleOpen ? `${consoleHeight}px` : "0" }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={stableNodeTypes}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onNodeClick={onNodeClick}
        onPaneClick={onPaneClick}
        fitView
        deleteKeyCode="Delete"
        className="bg-background"
      >
        <Controls />
        <MiniMap
          className="!bg-bg-secondary !border-border"
          maskColor="rgba(0,0,0,0.1)"
        />
        <Background variant={BackgroundVariant.Dots} gap={16} size={1} />
      </ReactFlow>

      {nodes.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <p className="text-sm text-text-secondary">
            Drag a card from the sidebar to get started
          </p>
        </div>
      )}
    </div>
  );
}
