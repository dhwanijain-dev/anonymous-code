'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from 'recharts';
import { ArrowLeft, Activity, Battery, MapPin, Calendar, AlertCircle, CheckCircle, TrendingUp } from 'lucide-react';
import { Sensor } from '@/lib/types';

// Mock data
const mockSensor: Sensor = {
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
};

const mockReadingsData = [
  { time: '00:00', value: 120 },
  { time: '04:00', value: 115 },
  { time: '08:00', value: 142 },
  { time: '12:00', value: 138 },
  { time: '16:00', value: 155 },
  { time: '20:00', value: 128 },
  { time: '23:59', value: 125 },
];

interface SensorDetailPageProps {
  params: Promise<{ id: string }>;
}

export default function SensorDetailPage({ params }: SensorDetailPageProps) {
  const router = useRouter();
  const [timeRange, setTimeRange] = useState<'24h' | '7d' | '30d'>('24h');

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Button
            variant="ghost"
            size="icon"
            onClick={() => router.back()}
          >
            <ArrowLeft className="w-5 h-5" />
          </Button>
          <div>
            <h1 className="text-3xl font-bold">{mockSensor.name}</h1>
            <p className="text-muted-foreground mt-1">{mockSensor.zone_id}</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {mockSensor.status === 'active' ? (
            <div className="flex items-center gap-2 px-3 py-1 bg-success/20 text-success rounded-full text-sm font-medium border border-success/30">
              <CheckCircle className="w-4 h-4" />
              Active
            </div>
          ) : (
            <div className="flex items-center gap-2 px-3 py-1 bg-destructive/20 text-destructive rounded-full text-sm font-medium border border-destructive/30">
              <AlertCircle className="w-4 h-4" />
              {mockSensor.status}
            </div>
          )}
        </div>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="pt-6">
            <p className="text-sm text-muted-foreground">Current Reading</p>
            <p className="text-2xl font-bold text-primary mt-2">
              {mockSensor.last_reading?.value}
              <span className="text-xs text-muted-foreground ml-1">{mockSensor.last_reading?.unit}</span>
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Battery Level</p>
                <p className="text-2xl font-bold mt-2">{mockSensor.battery_level}%</p>
              </div>
              <Battery className="w-8 h-8 text-warning" />
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <p className="text-sm text-muted-foreground">Sensor Type</p>
            <p className="text-lg font-bold capitalize mt-2">{mockSensor.sensor_type.replace('_', ' ')}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <p className="text-sm text-muted-foreground">Status</p>
            <div className="flex items-center gap-2 mt-2">
              <div className="w-3 h-3 bg-success rounded-full"></div>
              <span className="font-medium capitalize">{mockSensor.status}</span>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Readings over time */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Sensor Readings</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex gap-2 mb-4">
              {(['24h', '7d', '30d'] as const).map((range) => (
                <Button
                  key={range}
                  size="sm"
                  variant={timeRange === range ? 'default' : 'outline'}
                  onClick={() => setTimeRange(range)}
                >
                  {range}
                </Button>
              ))}
            </div>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={mockReadingsData}>
                <defs>
                  <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.8} />
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis dataKey="time" stroke="hsl(var(--muted-foreground))" />
                <YAxis stroke="hsl(var(--muted-foreground))" />
                <Tooltip />
                <Area type="monotone" dataKey="value" stroke="#3b82f6" fillOpacity={1} fill="url(#colorValue)" />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Statistics */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Statistics</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <p className="text-sm text-muted-foreground">Average Reading ({timeRange})</p>
              <p className="text-2xl font-bold">132.4 L/min</p>
            </div>
            <div className="space-y-2">
              <p className="text-sm text-muted-foreground">Peak Reading ({timeRange})</p>
              <p className="text-2xl font-bold">155 L/min</p>
            </div>
            <div className="space-y-2">
              <p className="text-sm text-muted-foreground">Minimum Reading ({timeRange})</p>
              <p className="text-2xl font-bold">115 L/min</p>
            </div>
            <div className="space-y-2">
              <p className="text-sm text-muted-foreground">Anomaly Count ({timeRange})</p>
              <p className="text-2xl font-bold text-warning">2</p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Sensor Information */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Sensor Information</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div>
              <p className="text-sm text-muted-foreground flex items-center gap-2">
                <MapPin className="w-4 h-4" />
                Coordinates
              </p>
              <p className="font-medium mt-2 text-sm">
                {mockSensor.coordinates.latitude.toFixed(4)}°N, {Math.abs(mockSensor.coordinates.longitude).toFixed(4)}°W
              </p>
            </div>
            <div>
              <p className="text-sm text-muted-foreground flex items-center gap-2">
                <Calendar className="w-4 h-4" />
                Installation Date
              </p>
              <p className="font-medium mt-2 text-sm">
                {new Date(mockSensor.installation_date).toLocaleDateString()}
              </p>
            </div>
            <div>
              <p className="text-sm text-muted-foreground flex items-center gap-2">
                <Activity className="w-4 h-4" />
                Last Updated
              </p>
              <p className="font-medium mt-2 text-sm">
                {new Date(mockSensor.updated_at).toLocaleDateString()} {new Date(mockSensor.updated_at).toLocaleTimeString()}
              </p>
            </div>
            <div>
              <p className="text-sm text-muted-foreground flex items-center gap-2">
                <TrendingUp className="w-4 h-4" />
                Uptime
              </p>
              <p className="font-medium mt-2 text-sm">99.8%</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
