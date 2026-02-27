'use client';

import { useMemo } from 'react';
import { Thermometer, AlertTriangle } from 'lucide-react';
import dynamic from 'next/dynamic';
import type { ComponentType } from 'react';

interface HeatmapData {
  nodes: string[];
  pressure: number[];
  riskNodes?: string[];
}

interface PressureHeatmapProps {
  data: HeatmapData | null;
  isLoading?: boolean;
}

interface HeatmapPoint {
  lat: number;
  lng: number;
  intensity: number;
  node: string;
  isRisk: boolean;
}

interface LeafletHeatmapInnerProps {
  points: HeatmapPoint[];
  minPressure: number;
  maxPressure: number;
}

// Dynamically import the inner Leaflet component to avoid SSR issues
const LeafletHeatmapInner = dynamic<LeafletHeatmapInnerProps>(
    //@ts-ignore
  () => import('./leaflet-heatmap-inner').then(mod => mod.default) as unknown as Promise<ComponentType<LeafletHeatmapInnerProps>>,
  {
    ssr: false,
    loading: () => (
      <div className="w-full h-full bg-linear-to-br from-blue-50 to-indigo-100 rounded-xl flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <div className="w-8 h-8 border-3 border-slate-600 border-t-transparent rounded-full animate-spin mx-auto mb-2" />
          <p className="text-sm text-slate-500">Loading map...</p>
        </div>
      </div>
    )
  }
);

// Seeded random number generator for deterministic but natural-looking results
const seededRandom = (seed: number) => {
  const x = Math.sin(seed) * 10000;
  return x - Math.floor(x);
};

// Generate realistic pipeline network coordinates
// Simulates a water distribution network with main lines and branches
const generateHeatmapPoints = (data: HeatmapData): HeatmapPoint[] => {
  const baseCoords = { lat: 22.719568, lng: 75.857727 }; // Indore, India
  const networkSpread = 0.015; // Wider area coverage
  
  // Create multiple pipeline "branches" for realistic distribution
  const branches = [
    { angle: 0, length: 1.0 },      // Main east-west line
    { angle: 45, length: 0.8 },     // NE diagonal
    { angle: 90, length: 0.9 },     // North-south line  
    { angle: 135, length: 0.7 },    // NW diagonal
    { angle: 180, length: 0.85 },   // West extension
    { angle: 225, length: 0.6 },    // SW branch
    { angle: 270, length: 0.75 },   // South branch
    { angle: 315, length: 0.65 },   // SE branch
  ];
  
  return data.nodes.map((node, index) => {
    // Use node name as seed for deterministic randomness
    const seed = node.split('').reduce((acc, char, i) => acc + char.charCodeAt(0) * (i + 1), 0);
    
    // Select a branch for this node
    const branchIndex = Math.floor(seededRandom(seed) * branches.length);
    const branch = branches[branchIndex];
    
    // Position along the branch (0 to 1)
    const positionAlongBranch = seededRandom(seed * 2) * branch.length;
    
    // Convert angle to radians
    const angleRad = (branch.angle * Math.PI) / 180;
    
    // Base position along the pipeline branch
    const baseLat = baseCoords.lat + Math.sin(angleRad) * positionAlongBranch * networkSpread;
    const baseLng = baseCoords.lng + Math.cos(angleRad) * positionAlongBranch * networkSpread;
    
    // Add organic variation perpendicular to the branch direction
    const perpAngle = angleRad + Math.PI / 2;
    const perpOffset = (seededRandom(seed * 3) - 0.5) * networkSpread * 0.3;
    
    // Add small random noise for natural clustering
    const noiseX = (seededRandom(seed * 5) - 0.5) * networkSpread * 0.15;
    const noiseY = (seededRandom(seed * 7) - 0.5) * networkSpread * 0.15;
    
    const finalLat = baseLat + Math.sin(perpAngle) * perpOffset + noiseY;
    const finalLng = baseLng + Math.cos(perpAngle) * perpOffset + noiseX;
    
    return {
      lat: finalLat,
      lng: finalLng,
      intensity: data.pressure[index] ?? 0,
      node: node,
      isRisk: data.riskNodes?.includes(node) ?? false,
    };
  });
};

export function PressureHeatmap({ data, isLoading }: PressureHeatmapProps) {
  const { points, minPressure, maxPressure, avgPressure } = useMemo(() => {
    if (!data || !data.nodes || !data.pressure || data.nodes.length === 0) {
      return { points: [], minPressure: 0, maxPressure: 100, avgPressure: 50 };
    }
    
    const pressures = data.pressure.filter(p => typeof p === 'number' && !isNaN(p));
    const min = pressures.length > 0 ? Math.min(...pressures) : 0;
    const max = pressures.length > 0 ? Math.max(...pressures) : 100;
    const avg = pressures.length > 0 ? pressures.reduce((a, b) => a + b, 0) / pressures.length : 50;
    
    return {
      points: generateHeatmapPoints(data),
      minPressure: min,
      maxPressure: max,
      avgPressure: avg,
    };
  }, [data]);

  if (isLoading || !data) {
    return (
      <div className="bg-white/80 backdrop-blur-sm rounded-2xl border border-slate-200 shadow-lg p-6 h-[calc(100vh-280px)] min-h-[500px]">
        <div className="flex items-center gap-3 mb-6">
          <div className="p-2 rounded-xl bg-linear-to-br from-rose-500 to-orange-500 shadow-lg">
            <Thermometer className="w-5 h-5 text-white" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-slate-900">Pipeline Pressure Heatmap</h3>
            <p className="text-sm text-slate-500">Real-time pressure distribution</p>
          </div>
        </div>
        <div className="h-full flex items-center justify-center">
          <div className="text-center">
            <div className="w-10 h-10 border-3 border-slate-600 border-t-transparent rounded-full animate-spin mx-auto mb-3" />
            <p className="text-sm text-slate-500">Loading heatmap data...</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white/80 backdrop-blur-sm rounded-2xl border border-slate-200 shadow-lg p-6 h-[calc(100vh-280px)] min-h-[500px] flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between mb-4 shrink-0">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-xl bg-linear-to-br from-rose-500 to-orange-500 shadow-lg">
            <Thermometer className="w-5 h-5 text-white" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-slate-900">Pipeline Pressure Heatmap</h3>
            <p className="text-sm text-slate-500">Real-time pressure distribution across {data.nodes.length} nodes</p>
          </div>
        </div>
        
        {/* Stats */}
        <div className="flex items-center gap-6 text-sm">
          <div className="text-center">
            <p className="text-slate-500">Min</p>
            <p className="font-semibold text-red-600">{minPressure.toFixed(1)} bar</p>
          </div>
          <div className="text-center">
            <p className="text-slate-500">Avg</p>
            <p className="font-semibold text-amber-600">{avgPressure.toFixed(1)} bar</p>
          </div>
          <div className="text-center">
            <p className="text-slate-500">Max</p>
            <p className="font-semibold text-emerald-600">{maxPressure.toFixed(1)} bar</p>
          </div>
        </div>
      </div>

      {/* Legend */}
      <div className="flex items-center justify-between mb-4 shrink-0">
        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-500">Low Pressure (Risk)</span>
          <div className="w-32 h-3 rounded-full" style={{
            background: 'linear-gradient(to right, #ef4444, #f59e0b, #eab308, #84cc16, #22c55e)'
          }} />
          <span className="text-xs text-slate-500">High Pressure (Safe)</span>
        </div>
        <div className="flex items-center gap-2 text-xs text-slate-500">
          <AlertTriangle className="w-3 h-3 text-red-500" />
          <span>Risk Node</span>
        </div>
      </div>

      {/* Map Container */}
      <div className="flex-1 rounded-xl overflow-hidden border border-slate-200">
        <LeafletHeatmapInner 
          points={points} 
          minPressure={minPressure} 
          maxPressure={maxPressure} 
        />
      </div>
    </div>
  );
}
