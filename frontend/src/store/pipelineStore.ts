import { create } from "zustand";
import { persist } from "zustand/middleware";
import {
  type Node,
  type Edge,
  type OnNodesChange,
  type OnEdgesChange,
  type OnConnect,
  applyNodeChanges,
  applyEdgeChanges,
  addEdge,
  type NodeChange,
} from "@xyflow/react";
import type {
  CardSchema,
  CardNodeData,
  NodeStatusValue,
  PipelineRequest,
  CardOutputPreview,
  JsonSchema,
} from "@/lib/types";

function extractDefaults(schema: JsonSchema): Record<string, unknown> {
  const defaults: Record<string, unknown> = {};
  if (schema.properties) {
    for (const [key, prop] of Object.entries(schema.properties)) {
      if (prop.default !== undefined) {
        defaults[key] = prop.default;
      }
    }
  }
  return defaults;
}

export interface LogEntry {
  timestamp: number;
  text: string;
}

interface PipelineState {
  nodes: Node<CardNodeData>[];
  edges: Edge[];
  nodeCounter: number;
  onNodesChange: OnNodesChange<Node<CardNodeData>>;
  onEdgesChange: OnEdgesChange;
  onConnect: OnConnect;

  cardSchemas: CardSchema[];
  setCardSchemas: (schemas: CardSchema[]) => void;

  addNode: (schema: CardSchema, position: { x: number; y: number }) => void;
  updateNodeConfig: (nodeId: string, config: Record<string, unknown>) => void;
  removeNode: (nodeId: string) => void;

  selectedNodeId: string | null;
  selectNode: (nodeId: string | null) => void;

  pipelineId: string | null;
  isExecuting: boolean;
  executingNodeIds: string[];
  validationErrors: string[];
  logEntries: LogEntry[];
  consoleOpen: boolean;
  consoleHeight: number;
  consoleActiveTab: "console" | "output" | "config";
  selectedNodeIdForOutput: string | null;
  setPipelineId: (id: string) => void;
  setIsExecuting: (executing: boolean) => void;
  setExecutingNodeIds: (ids: string[]) => void;
  setValidationErrors: (errors: string[]) => void;
  setNodeStatus: (nodeId: string, status: NodeStatusValue, error?: string | null) => void;
  resetAllStatuses: () => void;
  resetNodeStatuses: (nodeIds: string[]) => void;
  setNodeOutput: (nodeId: string, output: CardOutputPreview) => void;
  addLog: (text: string) => void;
  clearLogs: () => void;
  setConsoleOpen: (open: boolean) => void;
  setConsoleHeight: (height: number) => void;
  setConsoleActiveTab: (tab: "console" | "output" | "config") => void;
  setSelectedNodeIdForOutput: (nodeId: string | null) => void;

  activeView: "board" | "editor";
  setActiveView: (view: "board" | "editor") => void;

  getUpstreamNodeIds: (targetNodeId: string) => string[];
  toPipelineRequest: (pipelineId: string, nodeIds?: string[]) => PipelineRequest;
}

export const usePipelineStore = create<PipelineState>()(
  persist(
    (set, get) => ({
      nodes: [],
      edges: [],
      nodeCounter: 0,

      onNodesChange: (changes) => {
        set({ nodes: applyNodeChanges(changes, get().nodes) });
      },

      onEdgesChange: (changes) => {
        set({ edges: applyEdgeChanges(changes, get().edges) });
      },

  onConnect: (connection) => {
    const { source, target, sourceHandle, targetHandle } = connection;
    const nodes = get().nodes;
    const sourceNode = nodes.find((n) => n.id === source);
    const targetNode = nodes.find((n) => n.id === target);
    if (!sourceNode || !targetNode) return;

    const sourceOutputName = sourceHandle || "default";
    const sourceOutputType =
      sourceNode.data.cardSchema.output_schema[sourceOutputName];
    
    // If targetHandle is not set, try to find matching input by type
    let targetInputName = targetHandle || "default";
    if (!targetHandle) {
      const targetInputSchema = targetNode.data.cardSchema.input_schema;
      // Find first input that matches the output type
      const matchingInput = Object.entries(targetInputSchema).find(
        ([_, inputType]) => inputType === sourceOutputType
      );
      if (matchingInput) {
        targetInputName = matchingInput[0];
      }
    }

    const targetInputType =
      targetNode.data.cardSchema.input_schema[targetInputName];

    if (sourceOutputType !== targetInputType) return;

    set({
      edges: addEdge(
        {
          ...connection,
          id: `e-${source}-${sourceOutputName}-${target}-${targetInputName}`,
          label: sourceOutputName,
          data: {
            source_output: sourceOutputName,
            target_input: targetInputName,
          },
        },
        get().edges
      ),
    });
  },

  cardSchemas: [],
  setCardSchemas: (schemas) => set({ cardSchemas: schemas }),

  addNode: (schema, position) => {
    const counter = get().nodeCounter + 1;
    const id = `${schema.card_type}-${counter}`;
    const defaults = extractDefaults(schema.config_schema);
    const newNode: Node<CardNodeData> = {
      id,
      type: "cardNode",
      position,
      data: {
        label: schema.display_name,
        cardSchema: schema,
        config: defaults,
        status: "pending",
        error: null,
        outputPreview: null,
      },
    };
    set({ nodes: [...get().nodes, newNode], nodeCounter: counter });
  },

  updateNodeConfig: (nodeId, config) => {
    set({
      nodes: get().nodes.map((n) =>
        n.id === nodeId ? { ...n, data: { ...n.data, config } } : n
      ),
    });
  },

  removeNode: (nodeId) => {
    const { nodes, edges } = get();
    
    // Remove the node
    const nodeChanges: NodeChange[] = [{ type: "remove", id: nodeId }];
    const updatedNodes = applyNodeChanges(nodeChanges, nodes);
    
    // Remove connected edges
    const updatedEdges = edges.filter(
      (e) => e.source !== nodeId && e.target !== nodeId
    );
    
    set({ nodes: updatedNodes, edges: updatedEdges });
  },

  selectedNodeId: null,
  selectNode: (nodeId) => set({ selectedNodeId: nodeId }),

  pipelineId: null,
  isExecuting: false,
  executingNodeIds: [],
  validationErrors: [],
  logEntries: [],
  consoleOpen: false,
  consoleHeight: 300,
  consoleActiveTab: "console" as const,
  selectedNodeIdForOutput: null,
  activeView: "board" as const,
  setActiveView: (view) => set({ activeView: view }),
  setPipelineId: (id) => set({ pipelineId: id }),
  setIsExecuting: (executing) => set({ isExecuting: executing }),
  setExecutingNodeIds: (ids) => set({ executingNodeIds: ids }),
  setValidationErrors: (errors) => set({ validationErrors: errors }),
  addLog: (text) =>
    set({ logEntries: [...get().logEntries, { timestamp: Date.now(), text }] }),
  clearLogs: () => set({ logEntries: [] }),
  setConsoleOpen: (open) => set({ consoleOpen: open }),
  setConsoleHeight: (height) => set({ consoleHeight: Math.max(100, Math.min(600, height)) }),
  setConsoleActiveTab: (tab) => set({ consoleActiveTab: tab }),
  setSelectedNodeIdForOutput: (nodeId) => set({ selectedNodeIdForOutput: nodeId }),

  setNodeStatus: (nodeId, status, error = null) => {
    set({
      nodes: get().nodes.map((n) =>
        n.id === nodeId ? { ...n, data: { ...n.data, status, error } } : n
      ),
    });
  },

  resetAllStatuses: () => {
    set({
      nodes: get().nodes.map((n) => ({
        ...n,
        data: { ...n.data, status: "pending" as const, error: null, outputPreview: null },
      })),
    });
  },

  resetNodeStatuses: (nodeIds) => {
    const idSet = new Set(nodeIds);
    set({
      nodes: get().nodes.map((n) =>
        idSet.has(n.id)
          ? { ...n, data: { ...n.data, status: "pending" as const, error: null, outputPreview: null } }
          : n
      ),
    });
  },

  setNodeOutput: (nodeId, output) => {
    set({
      nodes: get().nodes.map((n) =>
        n.id === nodeId
          ? { ...n, data: { ...n.data, outputPreview: output } }
          : n
      ),
    });
  },

  getUpstreamNodeIds: (targetNodeId) => {
    const { edges } = get();
    const result = new Set<string>();
    const queue = [targetNodeId];
    while (queue.length > 0) {
      const current = queue.pop()!;
      if (result.has(current)) continue;
      result.add(current);
      for (const e of edges) {
        if (e.target === current && !result.has(e.source)) {
          queue.push(e.source);
        }
      }
    }
    return Array.from(result);
  },

  toPipelineRequest: (pipelineId, nodeIds) => {
    const { nodes, edges } = get();
    const idSet = nodeIds ? new Set(nodeIds) : null;
    const filteredNodes = idSet ? nodes.filter((n) => idSet.has(n.id)) : nodes;
    const filteredEdges = idSet
      ? edges.filter((e) => idSet.has(e.source) && idSet.has(e.target))
      : edges;
      return {
        pipeline_id: pipelineId,
        nodes: filteredNodes.map((n) => ({
          id: n.id,
          type: n.data.cardSchema.card_type,
          config: n.data.config,
          position: n.position,
        })),
      edges: filteredEdges.map((e) => {
        // Get source_output and target_input from edge data, or fallback to handles
        const sourceOutput = (e.data as Record<string, string>)?.source_output 
          || (e.sourceHandle as string) 
          || "default";
        const targetInput = (e.data as Record<string, string>)?.target_input 
          || (e.targetHandle as string) 
          || "default";
        
        return {
          source: e.source,
          target: e.target,
          source_output: sourceOutput,
          target_input: targetInput,
        };
      }),
      };
    },
  }),
  {
    name: "tensorrag-pipeline-storage",
    partialize: (state) => ({
      nodes: state.nodes,
      edges: state.edges,
      nodeCounter: state.nodeCounter,
      // Don't persist execution state, logs, or temporary data
    }),
  }
  )
);
