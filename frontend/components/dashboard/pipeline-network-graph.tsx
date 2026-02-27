'use client';

import { useCallback, useMemo, useEffect, useRef } from 'react';
import { Network, AlertTriangle, Droplet } from 'lucide-react';
import { MapContainer, TileLayer, CircleMarker, Polyline, Popup, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';

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

// The ReactFlow custom node isn't used for the Leaflet overlay; keep visual logic
function nodeColorFor(node: PipelineNodeData, mode: 'detection' | 'predictive') {
  const isLeaking = node.is_leaking === 1;
  const leakProb = node.pred_leak_prob ?? 0;
  if (mode === 'detection') return isLeaking ? '#ef4444' : '#10b981';
  if (leakProb > 0.7) return '#ef4444';
  if (leakProb > 0.5) return '#f59e0b';
  if (leakProb > 0.3) return '#facc15';
  return '#10b981';
}

// Inner component that uses useReactFlow hook
function PipelineNetworkGraphInner({ nodes: graphNodes, edges: graphEdges, isLoading, mode = 'detection' }: PipelineNetworkGraphProps) {
  // Center on Indore by default
  const indoreCenter: [number, number] = [22.7196, 75.8577];

  // Helper: convert node x/y to lat/lng.
  // If values look like lat/lng already (x in -180..180, y in -90..90) assume x=lon,y=lat
  // otherwise normalize positions into a small bbox around Indore.
  const toLatLng = (node: GraphNode) : [number, number] => {
    const x = node.x ?? 0;
    const y = node.y ?? 0;
    if (x >= -180 && x <= 180 && y >= -90 && y <= 90) {
      // assume x = lon, y = lat
      return [y, x];
    }

    // fallback: normalize using min/max across graph
    const xs = graphNodes.map(n => n.x ?? 0);
    const ys = graphNodes.map(n => n.y ?? 0);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);

    const nx = (x - minX) / (maxX - minX || 1);
    const ny = (y - minY) / (maxY - minY || 1);

    // map normalized coords into ~ +/- 0.06 degrees around Indore
    const lat = indoreCenter[0] + (ny - 0.5) * 0.12;
    const lng = indoreCenter[1] + (nx - 0.5) * 0.12;
    return [lat, lng];
  };

  if (isLoading) {
    return (
      <div className="h-[600px]">
        <div className="h-[550px] flex items-center justify-center">
          <div className="text-center">
            <div className="w-10 h-10 border-3 border-slate-600 border-t-transparent rounded-full animate-spin mx-auto mb-3" />
            <p className="text-sm text-slate-500">Loading network data...</p>
          </div>
        </div>
      </div>
    );
  }

  const stats = {
    totalNodes: graphNodes.length,
    totalPipes: graphEdges.length,
    leakingNodes: graphNodes.filter(n => n.is_leaking === 1).length,
    leakingPipes: graphEdges.filter(e => e.is_leaking === 1).length,
    highRiskNodes: graphNodes.filter(n => (n.pred_leak_prob ?? 0) > 0.7).length,
    mediumRiskNodes: graphNodes.filter(n => (n.pred_leak_prob ?? 0) > 0.5 && (n.pred_leak_prob ?? 0) <= 0.7).length,
    lowRiskNodes: graphNodes.filter(n => (n.pred_leak_prob ?? 0) > 0.3 && (n.pred_leak_prob ?? 0) <= 0.5).length,
    predictedLeaks: graphNodes.filter(n => n.pred_is_leaking === 1).length,
  };

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <div className="text-sm text-slate-600">
          {stats.totalNodes} nodes • {stats.totalPipes} connections
        </div>
        <div className="flex items-center gap-4 text-xs">
          <div className="flex items-center gap-1.5"><div className="w-3 h-3 rounded-full bg-emerald-500" /><span className="text-slate-600">Normal</span></div>
          <div className="flex items-center gap-1.5"><div className="w-3 h-3 rounded-full bg-red-500" /><span className="text-slate-600">Active Leak ({stats.leakingNodes})</span></div>
        </div>
      </div>

      <div className="h-[600px] rounded-xl overflow-hidden border border-slate-200 bg-gradient-to-br from-slate-50 to-slate-100">
        <MapContainer center={indoreCenter} zoom={13} style={{ height: '100%', width: '100%' }}>
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />

          {/* Render edges as polylines */}
          {graphEdges.map((edge, idx) => {
            const from = graphNodes.find(n => n.id === edge.from);
            const to = graphNodes.find(n => n.id === edge.to);
            if (!from || !to) return null;
            const latlngs: [number, number][] = [toLatLng(from), toLatLng(to)];
            const color = edge.is_leaking === 1 ? '#ef4444' : (edge.pred_leak_prob && edge.pred_leak_prob > 0.5 ? '#f59e0b' : '#94a3b8');
            const weight = edge.is_leaking === 1 ? 5 : (edge.pred_leak_prob && edge.pred_leak_prob > 0.5 ? 3 : 2);
            return <Polyline key={edge.id || idx} positions={latlngs} pathOptions={{ color, weight, opacity: 0.9 }} />;
          })}

          {/* Render nodes as circle markers */}
          {graphNodes.map((node) => {
            const [lat, lng] = toLatLng(node);
            const color = nodeColorFor({ label: node.id, pressure: node.pressure, is_leaking: node.is_leaking, pred_leak_prob: node.pred_leak_prob }, mode);
            return (
              <CircleMarker
                key={node.id}
                center={[lat, lng]}
                radius={8}
                pathOptions={{ color, fillColor: color, fillOpacity: 0.9 }}
              >
                <Popup>
                  <div className="text-sm">
                    <div className="font-semibold">{node.id}</div>
                    <div>Pressure: {node.pressure?.toFixed(2)} bar</div>
                    <div>Pred Leak Prob: {(node.pred_leak_prob ?? 0).toFixed(2)}</div>
                    <div>Actual Leak: {node.is_leaking === 1 ? 'Yes' : 'No'}</div>
                  </div>
                </Popup>
              </CircleMarker>
            );
          })}
        </MapContainer>
      </div>
    </div>
  );
}

// Export the inner component directly (Leaflet-based)
export function PipelineNetworkGraph(props: PipelineNetworkGraphProps) {
  return <PipelineNetworkGraphInner {...props} />;
}
