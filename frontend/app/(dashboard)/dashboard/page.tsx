'use client';

import { useState, useEffect } from 'react';
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { AlertCircle, TrendingUp, Droplet, Zap, Activity } from 'lucide-react';
import { DashboardMetrics, Alert } from '@/lib/types';
import { KPICard } from '@/components/dashboard/kpi-card';
import { AlertFeed } from '@/components/dashboard/alert-feed';
import { MiniMap } from '@/components/dashboard/mini-map';

// Mock data
const mockMetrics: DashboardMetrics = {
  total_water_loss: 4250,
  total_water_loss_percentage: 12.5,
  critical_alerts: 3,
  active_leaks: 7,
  sensor_health_percentage: 94,
  zones_with_issues: 2,
  avg_water_loss_24h: 3890,
};

const mockChartData = [
  { time: '00:00', loss: 2400, expected: 2210 },
  { time: '04:00', loss: 2210, expected: 2290 },
  { time: '08:00', loss: 2290, expected: 2000 },
  { time: '12:00', loss: 2000, expected: 2181 },
  { time: '16:00', loss: 2181, expected: 2500 },
  { time: '20:00', loss: 2500, expected: 2100 },
  { time: '23:59', loss: 2100, expected: 2100 },
];

const mockAnomalyData = [
  { name: 'Normal', value: 85, color: '#06b6d4' },
  { name: 'Anomaly', value: 10, color: '#f97316' },
  { name: 'Critical', value: 5, color: '#ef4444' },
];

const mockAlerts: Alert[] = [
  {
    id: '1',
    alert_type: 'leak_detected',
    severity: 'critical',
    status: 'active',
    description: 'High-pressure leak detected in Zone A-2',
    location: 'Main Pipeline - Sector 2',
    estimated_loss: 850,
    created_at: new Date(Date.now() - 5 * 60000).toISOString(),
  },
  {
    id: '2',
    alert_type: 'pressure_drop',
    severity: 'high',
    status: 'active',
    description: 'Unusual pressure drop detected',
    location: 'Secondary Line - Sector 5',
    created_at: new Date(Date.now() - 15 * 60000).toISOString(),
  },
  {
    id: '3',
    alert_type: 'flow_anomaly',
    severity: 'medium',
    status: 'acknowledged',
    description: 'Flow rate exceeds expected parameters',
    location: 'Zone C-1',
    created_at: new Date(Date.now() - 45 * 60000).toISOString(),
  },
];

export default function DashboardPage() {
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    // Simulate fetching data
    setIsLoading(false);
  }, []);

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-slate-900">Dashboard</h1>
        <p className="text-slate-500 mt-1">Real-time water leakage monitoring and analytics</p>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <KPICard
          title="Water Loss"
          value={`${mockMetrics.total_water_loss} m³`}
          change={mockMetrics.total_water_loss_percentage}
          unit="%"
          icon={Droplet}
          trend="up"
          status="warning"
        />
        <KPICard
          title="Critical Alerts"
          value={mockMetrics.critical_alerts}
          change={2}
          unit="active"
          icon={AlertCircle}
          trend="down"
          status="critical"
        />
        <KPICard
          title="Active Leaks"
          value={mockMetrics.active_leaks}
          change={1}
          unit="sites"
          icon={Zap}
          trend="up"
          status="warning"
        />
        <KPICard
          title="Sensor Health"
          value={`${mockMetrics.sensor_health_percentage}%`}
          change={2}
          unit="operational"
          icon={TrendingUp}
          trend="down"
          status="success"
        />
      </div>

      {/* Charts and Alerts */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Water Loss Trend */}
        <div className="lg:col-span-2 bg-white/80 backdrop-blur-sm rounded-2xl border border-slate-200 shadow-lg p-6">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h3 className="text-lg font-semibold text-slate-900">Water Loss Trend</h3>
              <p className="text-sm text-slate-500">Last 24 hours</p>
            </div>
            <div className="flex items-center gap-4 text-xs">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-orange-500" />
                <span className="text-slate-600">Actual Loss</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-slate-700" />
                <span className="text-slate-600">Expected</span>
              </div>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={mockChartData}>
              <defs>
                <linearGradient id="colorLoss" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#f97316" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#f97316" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="colorExpected" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#334155" stopOpacity={0.2} />
                  <stop offset="95%" stopColor="#334155" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.1)" />
              <XAxis dataKey="time" stroke="rgba(0,0,0,0.4)" fontSize={12} />
              <YAxis stroke="rgba(0,0,0,0.4)" fontSize={12} />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: 'white', 
                  border: '1px solid rgba(0,0,0,0.1)',
                  borderRadius: '12px',
                  color: '#0f172a',
                  boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                }}
              />
              <Area type="monotone" dataKey="loss" stroke="#f97316" strokeWidth={2} fillOpacity={1} fill="url(#colorLoss)" />
              <Area type="monotone" dataKey="expected" stroke="#334155" strokeWidth={2} fillOpacity={1} fill="url(#colorExpected)" />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Anomaly Distribution */}
        <div className="bg-white/80 backdrop-blur-sm rounded-2xl border border-slate-200 shadow-lg p-6">
          <h3 className="text-lg font-semibold text-slate-900 mb-6">Data Distribution</h3>
          <div className="flex items-center justify-center">
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie
                  data={mockAnomalyData}
                  cx="50%"
                  cy="50%"
                  innerRadius={50}
                  outerRadius={80}
                  paddingAngle={3}
                  dataKey="value"
                >
                  {mockAnomalyData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="space-y-2 mt-4">
            {mockAnomalyData.map((item) => (
              <div key={item.name} className="flex items-center justify-between text-sm">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full" style={{ backgroundColor: item.color }} />
                  <span className="text-slate-600">{item.name}</span>
                </div>
                <span className="text-slate-900 font-medium">{item.value}%</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Mini Map and Alert Feed */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <MiniMap />
        <div className="lg:col-span-2">
          <AlertFeed alerts={mockAlerts} />
        </div>
      </div>
    </div>
  );
}
