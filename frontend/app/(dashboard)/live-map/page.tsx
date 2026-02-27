'use client';

import { useState, useEffect, useCallback } from 'react';
import dynamic from 'next/dynamic';
import type { ComponentType } from 'react';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { MapPin, AlertCircle, Droplet, Activity, Battery, Radio, Search, RefreshCw } from 'lucide-react';
import { Sensor } from '@/lib/types';
import { cn } from '@/lib/utils';
import { backendService, DerivedSensor } from '@/lib/backend-service';

interface FullMapProps {
  sensors: Sensor[];
  selectedSensor: Sensor | null;
  onSensorSelect: (sensor: Sensor | null) => void;
  center?: [number, number];
  zoom?: number;
}

// Dynamically import map component to avoid SSR issues with Leaflet
const FullLeafletMap = dynamic<FullMapProps>(
  () => import('@/components/dashboard/full-leaflet-map') as Promise<{ default: ComponentType<FullMapProps> }>, 
  { 
    ssr: false,
    loading: () => (
      <div className="w-full h-full bg-linear-to-br from-blue-50 via-indigo-50 to-slate-100 flex items-center justify-center">
        <div className="text-center">
          <div className="w-10 h-10 border-3 border-slate-600 border-t-transparent rounded-full animate-spin mx-auto mb-3" />
          <p className="text-sm text-slate-500">Loading map...</p>
        </div>
      </div>
    )
  }
);

export default function LiveMapPage() {
  const [sensors, setSensors] = useState<Sensor[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedSensor, setSelectedSensor] = useState<Sensor | null>(null);
  const [searchTerm, setSearchTerm] = useState('');

  // Fetch sensors from backend
  const fetchSensors = useCallback(async () => {
    try {
      const backendSensors = await backendService.getSensors();
      setSensors(backendSensors as Sensor[]);
      setIsLoading(false);
    } catch (error) {
      console.error('Failed to fetch sensors for map:', error);
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchSensors();
    const interval = setInterval(fetchSensors, 5000);
    return () => clearInterval(interval);
  }, [fetchSensors]);

  const filteredSensors = sensors.filter(
    (sensor) =>
      sensor.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      sensor.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
      sensor.zone_id.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const getSensorIcon = (type: Sensor['sensor_type']) => {
    switch (type) {
      case 'flow':
        return <Activity className="w-4 h-4" />;
      case 'pressure':
        return <Droplet className="w-4 h-4" />;
      case 'leak_detection':
        return <AlertCircle className="w-4 h-4" />;
      case 'acoustic':
        return <Radio className="w-4 h-4" />;
    }
  };

  const getStatusColor = (status: Sensor['status']) => {
    switch (status) {
      case 'active':
        return 'bg-emerald-100 text-emerald-700 border-emerald-200';
      case 'inactive':
        return 'bg-slate-100 text-slate-500 border-slate-200';
      case 'error':
        return 'bg-red-100 text-red-700 border-red-200';
      case 'maintenance':
        return 'bg-amber-100 text-amber-700 border-amber-200';
    }
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-slate-900">Live Map</h1>
          <p className="text-slate-500 mt-1">Real-time sensor locations and status (Network nodes)</p>
        </div>
        <Button
          onClick={fetchSensors}
          variant="outline"
          size="sm"
          className="flex items-center gap-2"
        >
          <RefreshCw className={cn("w-4 h-4", isLoading && "animate-spin")} />
          Refresh
        </Button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Map area */}
        <div className="lg:col-span-2">
          <div className="bg-white/80 backdrop-blur-sm rounded-2xl border border-slate-200 shadow-lg h-96 overflow-hidden">
            <FullLeafletMap 
              sensors={sensors}
              selectedSensor={selectedSensor}
              onSensorSelect={setSelectedSensor}
            />
          </div>
        </div>

        {/* Sidebar */}
        <div className="space-y-4">
          {/* Search */}
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
            <Input
              placeholder="Search sensors..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-10 bg-slate-50 border-slate-200 text-slate-900 placeholder:text-slate-400 rounded-xl focus:border-slate-400"
            />
          </div>

          {/* Sensor list */}
          <div className="bg-white/80 backdrop-blur-sm rounded-2xl border border-slate-200 shadow-lg overflow-hidden">
            <div className="px-4 py-3 border-b border-slate-200">
              <h3 className="text-sm font-medium text-slate-900">Sensors ({filteredSensors.length})</h3>
            </div>
            <div className="p-2 space-y-2 max-h-[400px] overflow-y-auto">
              {filteredSensors.map((sensor) => (
                <div
                  key={sensor.id}
                  onClick={() => setSelectedSensor(sensor)}
                  className={cn(
                    'p-3 rounded-xl cursor-pointer transition-all duration-300',
                    selectedSensor?.id === sensor.id
                      ? 'bg-linear-to-r from-slate-100 to-blue-50 border border-slate-300'
                      : 'bg-slate-50 border border-transparent hover:bg-slate-100 hover:border-slate-200'
                  )}
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-slate-900 truncate">{sensor.name}</p>
                      <p className="text-xs text-slate-500">{sensor.zone_id}</p>
                    </div>
                    <div className={cn(
                      "p-1.5 rounded-lg",
                      sensor.last_reading?.is_anomaly 
                        ? "bg-amber-100 text-amber-600" 
                        : "bg-blue-100 text-blue-600"
                    )}>
                      {getSensorIcon(sensor.sensor_type)}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Selected sensor details */}
      {selectedSensor && (
        <div className="bg-white/80 backdrop-blur-sm rounded-2xl border border-slate-200 shadow-lg p-6">
          <h3 className="text-lg font-semibold text-slate-900 mb-6">Sensor Details</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="p-4 bg-slate-50 rounded-xl">
              <p className="text-sm text-slate-500">Sensor Name</p>
              <p className="font-medium text-slate-900 mt-1">{selectedSensor.name}</p>
            </div>
            <div className="p-4 bg-slate-50 rounded-xl">
              <p className="text-sm text-slate-500">Type</p>
              <p className="font-medium text-slate-900 mt-1 capitalize">{selectedSensor.sensor_type.replace('_', ' ')}</p>
            </div>
            <div className="p-4 bg-slate-50 rounded-xl">
              <p className="text-sm text-slate-500">Status</p>
              <div className={`inline-flex items-center gap-1.5 mt-1 px-2 py-1 rounded-lg text-xs font-medium border ${getStatusColor(selectedSensor.status)}`}>
                <span className="w-1.5 h-1.5 rounded-full bg-current"></span>
                {selectedSensor.status}
              </div>
            </div>
            <div className="p-4 bg-slate-50 rounded-xl">
              <p className="text-sm text-slate-500">Battery</p>
              <div className="flex items-center gap-2 mt-1">
                <Battery className={cn(
                  "w-4 h-4",
                  //@ts-ignore
                  selectedSensor.battery_level > 50 ? "text-emerald-600" : selectedSensor.battery_level > 20 ? "text-amber-600" : "text-red-600"
                )} />
                <span className="font-medium text-slate-900">{selectedSensor.battery_level}%</span>
              </div>
            </div>
            <div className="p-4 bg-slate-50 rounded-xl">
              <p className="text-sm text-slate-500">Current Reading</p>
              <p className={cn(
                "font-medium mt-1",
                selectedSensor.last_reading?.is_anomaly ? "text-amber-600" : "text-slate-900"
              )}>
                {selectedSensor.last_reading?.value} {selectedSensor.last_reading?.unit}
                {selectedSensor.last_reading?.is_anomaly && (
                  <span className="ml-2 text-xs bg-amber-100 text-amber-700 px-1.5 py-0.5 rounded">Anomaly</span>
                )}
              </p>
            </div>
            <div className="p-4 bg-slate-50 rounded-xl">
              <p className="text-sm text-slate-500">Coordinates</p>
              <p className="font-medium text-slate-900 mt-1 text-xs">
                {selectedSensor.coordinates.latitude.toFixed(4)}, {selectedSensor.coordinates.longitude.toFixed(4)}
              </p>
            </div>
            <div className="p-4 bg-slate-50 rounded-xl">
              <p className="text-sm text-slate-500">Installation Date</p>
              <p className="font-medium text-slate-900 mt-1 text-sm">{new Date(selectedSensor.installation_date).toLocaleDateString()}</p>
            </div>
            <div className="p-4 bg-slate-50 rounded-xl">
              <p className="text-sm text-slate-500">Last Updated</p>
              <p className="font-medium text-slate-900 mt-1 text-sm">{new Date(selectedSensor.updated_at).toLocaleDateString()}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
