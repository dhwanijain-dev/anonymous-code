'use client';

import { useEffect, useState } from 'react';
import { MapPin, AlertCircle, CheckCircle, Waves } from 'lucide-react';
import dynamic from 'next/dynamic';
import type { ComponentType } from 'react';

interface MapSensor {
  id: number;
  name: string;
  lat: number;
  lng: number;
  status: 'active' | 'warning' | 'error';
}

interface LeafletMapProps {
  sensors: MapSensor[];
}

// Dynamically import map component to avoid SSR issues with Leaflet
const LeafletMap = dynamic<LeafletMapProps>(
  () => import('./leaflet-map') as Promise<{ default: ComponentType<LeafletMapProps> }>, 
  { 
    ssr: false,
    loading: () => (
      <div className="w-full h-48 bg-linear-to-br from-blue-50 to-indigo-100 rounded-xl border border-slate-200 flex items-center justify-center">
        <div className="text-center">
          <div className="w-8 h-8 border-3 border-slate-600 border-t-transparent rounded-full animate-spin mx-auto mb-2" />
          <p className="text-sm text-slate-500">Loading map...</p>
        </div>
      </div>
    )
  }
);

const zones = [
  { name: 'Zone A', status: 'healthy', sensors: 3, icon: CheckCircle, color: 'text-emerald-600', lat: 40.7128, lng: -74.006 },
  { name: 'Zone B', status: 'warning', alerts: 2, icon: AlertCircle, color: 'text-amber-600', lat: 40.7138, lng: -74.016 },
  { name: 'Zone C', status: 'healthy', sensors: 5, icon: CheckCircle, color: 'text-emerald-600', lat: 40.7118, lng: -73.996 },
];

const sensors: MapSensor[] = [
  { id: 1, name: 'Sensor A1', lat: 40.7128, lng: -74.006, status: 'active' },
  { id: 2, name: 'Sensor A2', lat: 40.7135, lng: -74.008, status: 'active' },
  { id: 3, name: 'Sensor B1', lat: 40.7138, lng: -74.016, status: 'warning' },
  { id: 4, name: 'Sensor B2', lat: 40.7142, lng: -74.012, status: 'warning' },
  { id: 5, name: 'Sensor C1', lat: 40.7118, lng: -73.996, status: 'active' },
  { id: 6, name: 'Sensor C2', lat: 40.7115, lng: -73.999, status: 'active' },
];

export function MiniMap() {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  return (
    <div className="bg-white/80 backdrop-blur-sm rounded-2xl border border-slate-200 shadow-lg p-6">
      <h3 className="text-lg font-semibold text-slate-900 mb-6">Zone Overview</h3>
      
      <div className="space-y-4">
        {/* Leaflet Map */}
        <div className="relative w-full h-48 rounded-xl border border-slate-200 overflow-hidden">
          {mounted && <LeafletMap sensors={sensors} />}
        </div>

        {/* Zone status */}
        <div className="space-y-2">
          {zones.map((zone) => {
            const Icon = zone.icon;
            return (
              <div 
                key={zone.name}
                className="flex items-center justify-between p-3 bg-slate-50 hover:bg-slate-100 rounded-xl border border-slate-200 transition-all duration-300 cursor-pointer group"
              >
                <div className="flex items-center gap-3">
                  <Icon className={`w-4 h-4 ${zone.color}`} />
                  <span className="text-sm font-medium text-slate-900">{zone.name}</span>
                </div>
                <span className="text-xs text-slate-500 group-hover:text-slate-700 transition-colors">
                  {zone.sensors ? `${zone.sensors} sensors` : `${zone.alerts} alerts`}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
