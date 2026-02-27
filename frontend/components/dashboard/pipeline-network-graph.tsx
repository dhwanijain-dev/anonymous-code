'use client';

import { useCallback, useMemo, useEffect } from 'react';
import {
  ReactFlow,
  Node,
  Edge,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  MarkerType,
  Position,
  Handle,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { Network, AlertTriangle, Droplet } from 'lucide-react';

// Types for the graph data from backend
interface GraphNode {
  id: string;
  x: number;
  y: number;
  pressure: number;
  is_leaking: number;
  pred_leak_prob?: number;
  pred_is_leaking?: number;
}

interface GraphEdge {
  id: string;
  from: string;
  to: string;
  type: string;
  length?: number;
  diameter?: number;
  flowrate?: number;
  is_leaking: number;
  pred_leak_prob?: number;
  pred_is_leaking?: number;
}

interface PipelineNetworkGraphProps {
  nodes: GraphNode[];
  edges: GraphEdge[];
  isLoading?: boolean;
  mode?: 'detection' | 'predictive';
}

// Custom node data type
interface PipelineNodeData {
  label: string;
  pressure: number;
  is_leaking: number;
  pred_leak_prob?: number;
  pred_is_leaking?: number;
  mode?: 'detection' | 'predictive';
}

// Custom node component for pipeline junctions
function PipelineNode({ data }: { data: PipelineNodeData }) {
  const isLeaking = data.is_leaking === 1;
  const isPredictedLeak = data.pred_is_leaking === 1;
  const leakProb = data.pred_leak_prob ?? 0;
  const mode = data.mode || 'detection';
  
  // Color based on mode and leak status
  let bgColor = 'bg-emerald-500';
  let ringColor = 'ring-emerald-300';
  let showAlert = false;
  
  if (mode === 'detection') {
    // Detection mode: highlight actual leaks
    if (isLeaking) {
      bgColor = 'bg-red-500';
      ringColor = 'ring-red-300';
      showAlert = true;
    }
  } else {
    // Predictive mode: highlight based on prediction probability
    if (leakProb > 0.7) {
      bgColor = 'bg-red-500';
      ringColor = 'ring-red-300';
      showAlert = true;
    } else if (leakProb > 0.5 || isPredictedLeak) {
      bgColor = 'bg-amber-500';
      ringColor = 'ring-amber-300';
    } else if (leakProb > 0.3) {
      bgColor = 'bg-yellow-500';
      ringColor = 'ring-yellow-300';
    }
  }

  return (
    <div className="relative">
      <Handle type="target" position={Position.Top} className="bg-slate-400! w-2! h-2!" />
      <Handle type="target" position={Position.Left} className="bg-slate-400! w-2! h-2!" />
      
      <div 
        className={`
          w-8 h-8 rounded-full ${bgColor} 
          flex items-center justify-center
          ring-2 ${ringColor} ring-offset-1 ring-offset-white
          transition-all duration-300 cursor-pointer
          hover:scale-110 hover:shadow-lg
          ${showAlert ? 'animate-pulse' : ''}
        `}
        title={`${data.label}\nPressure: ${data.pressure?.toFixed(2)} bar\nLeak Prob: ${(leakProb * 100).toFixed(1)}%\nActual Leak: ${isLeaking ? 'Yes' : 'No'}`}
      >
        {showAlert ? (
          <AlertTriangle className="w-4 h-4 text-white" />
        ) : (
          <Droplet className="w-3 h-3 text-white" />
        )}
      </div>
      
      {/* Node label */}
      <div className="absolute -bottom-5 left-1/2 -translate-x-1/2 text-[8px] font-medium text-slate-600 whitespace-nowrap">
        {data.label?.toString().substring(0, 8)}
      </div>
      
      <Handle type="source" position={Position.Bottom} className="bg-slate-400! w-2! h-2!" />
      <Handle type="source" position={Position.Right} className="bg-slate-400! w-2! h-2!" />
    </div>
  );
}

const nodeTypes = {
  pipeline: PipelineNode,
};

export function PipelineNetworkGraph({ nodes: graphNodes, edges: graphEdges, isLoading, mode = 'detection' }: PipelineNetworkGraphProps) {
  // Transform backend data to React Flow format
  const { flowNodes, flowEdges, bounds } = useMemo(() => {
    if (!graphNodes || graphNodes.length === 0) {
      return { flowNodes: [], flowEdges: [], bounds: { minX: 0, maxX: 100, minY: 0, maxY: 100 } };
    }

    // Calculate bounds from actual coordinates
    const xCoords = graphNodes.map(n => n.x);
    const yCoords = graphNodes.map(n => n.y);
    const minX = Math.min(...xCoords);
    const maxX = Math.max(...xCoords);
    const minY = Math.min(...yCoords);
    const maxY = Math.max(...yCoords);
    
    // Scale factor to fit in viewport
    const padding = 50;
    const viewWidth = 800;
    const viewHeight = 500;
    
    const scaleX = (maxX - minX) > 0 ? (viewWidth - 2 * padding) / (maxX - minX) : 1;
    const scaleY = (maxY - minY) > 0 ? (viewHeight - 2 * padding) / (maxY - minY) : 1;
    const scale = Math.min(scaleX, scaleY, 1.5); // Cap scale to prevent huge nodes

    // Convert nodes
    const nodes: Node[] = graphNodes.slice(0, 100).map((node, index) => ({
      id: node.id,
      type: 'pipeline',
      position: {
        x: padding + (node.x - minX) * scale,
        y: padding + (node.y - minY) * scale,
      },
      data: {
        label: node.id,
        pressure: node.pressure,
        is_leaking: node.is_leaking,
        pred_leak_prob: node.pred_leak_prob,
        pred_is_leaking: node.pred_is_leaking,
        mode,
      },
    }));

    // Create node ID set for validation
    const nodeIds = new Set(nodes.map(n => n.id));

    // Convert edges (only include edges where both nodes exist)
    const edges: Edge[] = graphEdges
      .filter(edge => nodeIds.has(edge.from) && nodeIds.has(edge.to))
      .slice(0, 150)
      .map((edge, index) => {
        const isLeaking = edge.is_leaking === 1;
        const isPredictedLeak = edge.pred_is_leaking === 1;
        const leakProb = edge.pred_leak_prob ?? 0;
        
        let strokeColor = '#94a3b8'; // slate-400
        let strokeWidth = 2;
        let animated = false;
        
        if (mode === 'detection') {
          // Detection mode: highlight actual leaks
          if (isLeaking) {
            strokeColor = '#ef4444'; // red-500
            strokeWidth = 4;
            animated = true;
          } else if (edge.flowrate && Math.abs(edge.flowrate) > 0.01) {
            strokeColor = '#0ea5e9'; // sky-500
            animated = true;
          }
        } else {
          // Predictive mode: highlight based on prediction
          if (leakProb > 0.7) {
            strokeColor = '#ef4444'; // red-500
            strokeWidth = 4;
            animated = true;
          } else if (isPredictedLeak || leakProb > 0.5) {
            strokeColor = '#f59e0b'; // amber-500
            strokeWidth = 3;
            animated = true;
          } else if (leakProb > 0.3) {
            strokeColor = '#eab308'; // yellow-500
            strokeWidth = 2;
          } else if (edge.flowrate && Math.abs(edge.flowrate) > 0.01) {
            strokeColor = '#0ea5e9'; // sky-500
            animated = true;
          }
        }

        return {
          id: edge.id || `edge-${index}`,
          source: edge.from,
          target: edge.to,
          type: 'default',
          animated,
          style: {
            stroke: strokeColor,
            strokeWidth,
          },
          markerEnd: {
            type: MarkerType.ArrowClosed,
            color: strokeColor,
            width: 15,
            height: 15,
          },
          label: edge.type !== 'Pipe' ? edge.type : undefined,
          labelStyle: { fontSize: 8, fill: '#64748b' },
        };
      });

    return { flowNodes: nodes, flowEdges: edges, bounds: { minX, maxX, minY, maxY } };
  }, [graphNodes, graphEdges, mode]);

  const [nodes, setNodes, onNodesChange] = useNodesState(flowNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(flowEdges);

  // Update nodes and edges when data changes
  useEffect(() => {
    setNodes(flowNodes);
    setEdges(flowEdges);
  }, [flowNodes, flowEdges, setNodes, setEdges]);

  // Stats
  const stats = useMemo(() => {
    const leakingNodes = graphNodes.filter(n => n.is_leaking === 1).length;
    const predictedLeaks = graphNodes.filter(n => n.pred_is_leaking === 1).length;
    const highRiskNodes = graphNodes.filter(n => (n.pred_leak_prob ?? 0) > 0.7).length;
    const mediumRiskNodes = graphNodes.filter(n => (n.pred_leak_prob ?? 0) > 0.5 && (n.pred_leak_prob ?? 0) <= 0.7).length;
    const lowRiskNodes = graphNodes.filter(n => (n.pred_leak_prob ?? 0) > 0.3 && (n.pred_leak_prob ?? 0) <= 0.5).length;
    const leakingPipes = graphEdges.filter(e => e.is_leaking === 1).length;
    const highRiskPipes = graphEdges.filter(e => (e.pred_leak_prob ?? 0) > 0.5).length;
    return { 
      leakingNodes, 
      predictedLeaks, 
      leakingPipes, 
      totalNodes: graphNodes.length, 
      totalPipes: graphEdges.length,
      highRiskNodes,
      mediumRiskNodes,
      lowRiskNodes,
      highRiskPipes
    };
  }, [graphNodes, graphEdges]);

  if (isLoading) {
    return (
      <div className="h-[500px]">
        <div className="h-[450px] flex items-center justify-center">
          <div className="text-center">
            <div className="w-10 h-10 border-3 border-slate-600 border-t-transparent rounded-full animate-spin mx-auto mb-3" />
            <p className="text-sm text-slate-500">Loading network data...</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div>
      {/* Header with Legend */}
      <div className="flex items-center justify-between mb-4">
        <div className="text-sm text-slate-600">
          {stats.totalNodes} nodes • {stats.totalPipes} connections
        </div>

        {/* Legend - changes based on mode */}
        {mode === 'detection' ? (
          <div className="flex items-center gap-4 text-xs">
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-3 rounded-full bg-emerald-500" />
              <span className="text-slate-600">Normal</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-3 rounded-full bg-red-500 animate-pulse" />
              <span className="text-slate-600">Active Leak ({stats.leakingNodes})</span>
            </div>
          </div>
        ) : (
          <div className="flex items-center gap-4 text-xs">
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-3 rounded-full bg-emerald-500" />
              <span className="text-slate-600">Low Risk</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-3 rounded-full bg-yellow-500" />
              <span className="text-slate-600">Moderate ({stats.lowRiskNodes})</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-3 rounded-full bg-amber-500" />
              <span className="text-slate-600">High Risk ({stats.mediumRiskNodes})</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-3 rounded-full bg-red-500 animate-pulse" />
              <span className="text-slate-600">Critical ({stats.highRiskNodes})</span>
            </div>
          </div>
        )}
      </div>

      {/* Stats bar - changes based on mode */}
      {mode === 'detection' ? (
        <div className="flex items-center gap-6 mb-4 text-sm">
          <div className="px-3 py-1.5 rounded-lg bg-red-50 border border-red-200">
            <span className="text-slate-500">Active Leaks:</span>
            <span className="ml-2 font-semibold text-red-600">{stats.leakingNodes + stats.leakingPipes}</span>
          </div>
          <div className="px-3 py-1.5 rounded-lg bg-slate-100">
            <span className="text-slate-500">Leak Nodes:</span>
            <span className="ml-2 font-semibold text-slate-700">{stats.leakingNodes}</span>
          </div>
          <div className="px-3 py-1.5 rounded-lg bg-slate-100">
            <span className="text-slate-500">Leak Pipes:</span>
            <span className="ml-2 font-semibold text-slate-700">{stats.leakingPipes}</span>
          </div>
        </div>
      ) : (
        <div className="flex items-center gap-6 mb-4 text-sm">
          <div className="px-3 py-1.5 rounded-lg bg-red-50 border border-red-200">
            <span className="text-slate-500">Critical Risk:</span>
            <span className="ml-2 font-semibold text-red-600">{stats.highRiskNodes}</span>
          </div>
          <div className="px-3 py-1.5 rounded-lg bg-amber-50 border border-amber-200">
            <span className="text-slate-500">High Risk:</span>
            <span className="ml-2 font-semibold text-amber-600">{stats.mediumRiskNodes}</span>
          </div>
          <div className="px-3 py-1.5 rounded-lg bg-yellow-50 border border-yellow-200">
            <span className="text-slate-500">Moderate Risk:</span>
            <span className="ml-2 font-semibold text-yellow-600">{stats.lowRiskNodes}</span>
          </div>
          <div className="px-3 py-1.5 rounded-lg bg-slate-100">
            <span className="text-slate-500">Predicted Risks:</span>
            <span className="ml-2 font-semibold text-slate-700">{stats.predictedLeaks}</span>
          </div>
        </div>
      )}

      {/* Graph */}
      <div className="h-[450px] rounded-xl overflow-hidden border border-slate-200 bg-gradient-to-br from-slate-50 to-slate-100">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          nodeTypes={nodeTypes}
          fitView
          fitViewOptions={{ padding: 0.2 }}
          minZoom={0.1}
          maxZoom={2}
          defaultEdgeOptions={{
            type: 'default',
          }}
        >
          <Background color="#e2e8f0" gap={20} />
          
          <Controls 
            position="bottom-right"
            style={{ 
              display: 'flex', 
              flexDirection: 'row',
              gap: '4px',
              background: 'white',
              borderRadius: '8px',
              padding: '4px',
              boxShadow: '0 2px 10px rgba(0,0,0,0.1)'
            }}
          />
        </ReactFlow>
      </div>
    </div>
  );
}
