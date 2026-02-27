'use client';

import { useState } from 'react';
import Link from 'next/link';
import { Input } from '@/components/ui/input';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Activity, Droplet, AlertCircle, MapPin, Battery, ChevronRight, Search, Cpu } from 'lucide-react';
import { Sensor } from '@/lib/types';
import { cn } from '@/lib/utils';

// Mock data
const mockSensors: Sensor[] = [
  {
    id: '1',
    zone_id: 'zone-a',
    name: 'Main Inlet - A1',
    sensor_type: 'flow',
    coordinates: { latitude: 40.7128, longitude: -74.006 },
    status: 'active',
    battery_level: 92,
    installation_date: '2024-01-15',
    created_at: '2024-01-15',
    updated_at: '2024-02-26',
    last_reading: {
      id: '1',
      sensor_id: '1',
      value: 125.4,
      unit: 'L/min',
      timestamp: new Date().toISOString(),
      is_anomaly: false,
    },
  },
  {
    id: '2',
    zone_id: 'zone-b',
    name: 'Secondary Line - B2',
    sensor_type: 'pressure',
    coordinates: { latitude: 40.7138, longitude: -74.016 },
    status: 'active',
    battery_level: 78,
    installation_date: '2024-01-20',
    created_at: '2024-01-20',
    updated_at: '2024-02-26',
    last_reading: {
      id: '2',
      sensor_id: '2',
      value: 3.8,
      unit: 'bar',
      timestamp: new Date().toISOString(),
      is_anomaly: true,
    },
  },
  {
    id: '3',
    zone_id: 'zone-c',
    name: 'Leak Detection - C1',
    sensor_type: 'leak_detection',
    coordinates: { latitude: 40.7118, longitude: -73.996 },
    status: 'active',
    battery_level: 85,
    installation_date: '2024-02-01',
    created_at: '2024-02-01',
    updated_at: '2024-02-26',
    last_reading: {
      id: '3',
      sensor_id: '3',
      value: 0,
      unit: 'status',
      timestamp: new Date().toISOString(),
      is_anomaly: false,
    },
  },
  {
    id: '4',
    zone_id: 'zone-a',
    name: 'Pressure Monitor - A3',
    sensor_type: 'pressure',
    coordinates: { latitude: 40.7108, longitude: -74.026 },
    status: 'active',
    battery_level: 65,
    installation_date: '2024-01-25',
    created_at: '2024-01-25',
    updated_at: '2024-02-26',
    last_reading: {
      id: '4',
      sensor_id: '4',
      value: 4.2,
      unit: 'bar',
      timestamp: new Date().toISOString(),
      is_anomaly: false,
    },
  },
];

export default function SensorsPage() {
  const [searchTerm, setSearchTerm] = useState('');
  const [filterType, setFilterType] = useState<string>('all');
  const [filterStatus, setFilterStatus] = useState<string>('all');

  const filteredSensors = mockSensors.filter((sensor) => {
    const matchSearch = sensor.name.toLowerCase().includes(searchTerm.toLowerCase());
    const matchType = filterType === 'all' || sensor.sensor_type === filterType;
    const matchStatus = filterStatus === 'all' || sensor.status === filterStatus;
    return matchSearch && matchType && matchStatus;
  });

  const getSensorIcon = (type: Sensor['sensor_type']) => {
    switch (type) {
      case 'flow':
        return <Activity className="w-5 h-5" />;
      case 'pressure':
        return <Droplet className="w-5 h-5" />;
      case 'leak_detection':
        return <AlertCircle className="w-5 h-5" />;
      case 'acoustic':
        return <Cpu className="w-5 h-5" />;
    }
  };

  const getStatusStyles = (status: Sensor['status']) => {
    switch (status) {
      case 'active':
        return { bg: 'bg-emerald-100', text: 'text-emerald-700', border: 'border-emerald-200', dot: 'bg-emerald-500' };
      case 'inactive':
        return { bg: 'bg-slate-100', text: 'text-slate-500', border: 'border-slate-200', dot: 'bg-slate-400' };
      case 'error':
        return { bg: 'bg-red-100', text: 'text-red-700', border: 'border-red-200', dot: 'bg-red-500' };
      case 'maintenance':
        return { bg: 'bg-amber-100', text: 'text-amber-700', border: 'border-amber-200', dot: 'bg-amber-500' };
    }
  };

  const getBatteryColor = (level?: number) => {
    if (!level) return 'text-slate-400';
    if (level >= 80) return 'text-emerald-600';
    if (level >= 50) return 'text-amber-600';
    return 'text-red-600';
  };

  const getBatteryBg = (level?: number) => {
    if (!level) return 'bg-slate-100';
    if (level >= 80) return 'bg-emerald-100';
    if (level >= 50) return 'bg-amber-100';
    return 'bg-red-100';
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-slate-900">Sensors</h1>
          <p className="text-slate-500 mt-1">Manage and monitor all sensors in the network</p>
        </div>
        <div className="flex items-center gap-2 px-4 py-2 bg-blue-50 text-blue-700 rounded-xl text-sm font-medium border border-blue-200">
          <Cpu className="w-4 h-4" />
          {mockSensors.filter((s) => s.status === 'active').length} Online
        </div>
      </div>

      {/* Filters */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="relative">
          <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
          <Input
            placeholder="Search sensors..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-11 h-11 bg-slate-50 border-slate-200 text-slate-900 placeholder:text-slate-400 rounded-xl focus:bg-white focus:border-slate-400"
          />
        </div>
        <Select value={filterType} onValueChange={setFilterType}>
          <SelectTrigger className="h-11 bg-slate-50 border-slate-200 text-slate-900 rounded-xl">
            <SelectValue placeholder="Filter by type" />
          </SelectTrigger>
          <SelectContent className="bg-white border-slate-200 shadow-lg">
            <SelectItem value="all">All Types</SelectItem>
            <SelectItem value="flow">Flow Sensors</SelectItem>
            <SelectItem value="pressure">Pressure Sensors</SelectItem>
            <SelectItem value="leak_detection">Leak Detection</SelectItem>
            <SelectItem value="acoustic">Acoustic Sensors</SelectItem>
          </SelectContent>
        </Select>
        <Select value={filterStatus} onValueChange={setFilterStatus}>
          <SelectTrigger className="h-11 bg-slate-50 border-slate-200 text-slate-900 rounded-xl">
            <SelectValue placeholder="Filter by status" />
          </SelectTrigger>
          <SelectContent className="bg-white border-slate-200 shadow-lg">
            <SelectItem value="all">All Status</SelectItem>
            <SelectItem value="active">Active</SelectItem>
            <SelectItem value="inactive">Inactive</SelectItem>
            <SelectItem value="error">Error</SelectItem>
            <SelectItem value="maintenance">Maintenance</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* Sensors Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {filteredSensors.map((sensor) => {
          const statusStyles = getStatusStyles(sensor.status);
          return (
            <Link key={sensor.id} href={`/sensors/${sensor.id}`}>
              <div className="h-full bg-white/80 backdrop-blur-sm hover:bg-white rounded-2xl border border-slate-200 hover:border-slate-300 shadow-sm hover:shadow-md transition-all duration-300 cursor-pointer group p-6">
                <div className="space-y-4">
                  {/* Header */}
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <h3 className="font-semibold text-base text-slate-900 group-hover:text-slate-700">{sensor.name}</h3>
                      <p className="text-xs text-slate-500">{sensor.zone_id}</p>
                    </div>
                    <div className="p-2 rounded-xl bg-gradient-to-br from-slate-100 to-slate-200 border border-slate-200 text-slate-700">
                      {getSensorIcon(sensor.sensor_type)}
                    </div>
                  </div>

                  {/* Status badge */}
                  <div className="flex items-center gap-2">
                    <span className={cn(
                      'inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium border',
                      statusStyles.bg, statusStyles.text, statusStyles.border
                    )}>
                      <span className={cn('w-1.5 h-1.5 rounded-full animate-pulse', statusStyles.dot)} />
                      {sensor.status}
                    </span>
                  </div>

                  {/* Reading */}
                  <div className="bg-slate-50 rounded-xl p-4 border border-slate-100">
                    <p className="text-xs text-slate-500">Current Reading</p>
                    <p className="text-2xl font-bold text-slate-900 mt-1">
                      {sensor.last_reading?.value}
                      <span className="text-sm ml-1.5 text-slate-500">{sensor.last_reading?.unit}</span>
                    </p>
                    {sensor.last_reading?.is_anomaly && (
                      <span className="inline-block mt-2 text-xs px-2 py-0.5 bg-amber-100 text-amber-700 rounded-lg">Anomaly Detected</span>
                    )}
                  </div>

                  {/* Details */}
                  <div className="space-y-2 text-sm">
                    <div className="flex items-center justify-between py-2 border-b border-slate-100">
                      <span className="text-slate-500">Type</span>
                      <span className="font-medium text-slate-900 capitalize">{sensor.sensor_type.replace('_', ' ')}</span>
                    </div>
                    <div className="flex items-center justify-between py-2">
                      <span className="text-slate-500">Battery</span>
                      <div className="flex items-center gap-2">
                        <div className={cn('px-2 py-0.5 rounded-lg', getBatteryBg(sensor.battery_level))}>
                          <Battery className={cn('w-4 h-4', getBatteryColor(sensor.battery_level))} />
                        </div>
                        <span className={cn('font-medium', getBatteryColor(sensor.battery_level))}>{sensor.battery_level}%</span>
                      </div>
                    </div>
                  </div>

                  {/* Footer */}
                  <div className="flex items-center justify-between pt-3 border-t border-slate-100">
                    <span className="text-xs text-slate-400 group-hover:text-slate-600 transition-colors">View details</span>
                    <ChevronRight className="w-4 h-4 text-slate-300 group-hover:text-slate-500 group-hover:translate-x-1 transition-all" />
                  </div>
                </div>
              </div>
            </Link>
          );
        })}
      </div>

      {filteredSensors.length === 0 && (
        <div className="bg-white/80 backdrop-blur-sm rounded-2xl border border-slate-200 shadow-lg p-12 text-center">
          <MapPin className="w-12 h-12 text-slate-300 mx-auto mb-3" />
          <p className="text-slate-500">No sensors found matching your filters</p>
        </div>
      )}
    </div>
  );
}
