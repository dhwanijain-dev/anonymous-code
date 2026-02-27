'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  AreaChart,
  Area,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { TrendingUp, Calendar, Download, BarChart3, Activity, AlertTriangle } from 'lucide-react';
import { cn } from '@/lib/utils';

// Mock data
const waterLossData = [
  { date: 'Feb 1', loss: 2400, expected: 2000 },
  { date: 'Feb 3', loss: 2210, expected: 2100 },
  { date: 'Feb 5', loss: 2290, expected: 2150 },
  { date: 'Feb 7', loss: 2000, expected: 2000 },
  { date: 'Feb 9', loss: 2181, expected: 2050 },
  { date: 'Feb 11', loss: 2500, expected: 2200 },
  { date: 'Feb 13', loss: 2100, expected: 2100 },
  { date: 'Feb 15', loss: 2300, expected: 2150 },
  { date: 'Feb 17', loss: 2600, expected: 2250 },
  { date: 'Feb 19', loss: 2400, expected: 2100 },
  { date: 'Feb 21', loss: 2200, expected: 2050 },
  { date: 'Feb 23', loss: 2450, expected: 2200 },
  { date: 'Feb 25', loss: 2350, expected: 2100 },
];

const zoneComparisonData = [
  { name: 'Zone A', waterLoss: 4200, pressure: 3.8, flowRate: 125 },
  { name: 'Zone B', waterLoss: 3100, pressure: 4.1, flowRate: 142 },
  { name: 'Zone C', waterLoss: 2800, pressure: 3.5, flowRate: 98 },
  { name: 'Zone D', waterLoss: 3500, pressure: 3.9, flowRate: 112 },
];

const anomalyData = [
  { x: 1, y: 120, type: 'Normal' },
  { x: 2, y: 115, type: 'Normal' },
  { x: 3, y: 125, type: 'Normal' },
  { x: 4, y: 128, type: 'Normal' },
  { x: 5, y: 145, type: 'Anomaly' },
  { x: 6, y: 142, type: 'Anomaly' },
  { x: 7, y: 130, type: 'Normal' },
  { x: 8, y: 155, type: 'Anomaly' },
  { x: 9, y: 125, type: 'Normal' },
  { x: 10, y: 160, type: 'Anomaly' },
];

export default function AnalyticsPage() {
  const [timeRange, setTimeRange] = useState<'7d' | '30d' | '90d'>('30d');

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-slate-900">Analytics</h1>
          <p className="text-slate-500 mt-1">Water loss trends and system analysis</p>
        </div>
        <Button className="bg-slate-100 border border-slate-200 text-slate-700 hover:bg-slate-200 rounded-xl shadow-sm">
          <Download className="w-4 h-4 mr-2" />
          Export Report
        </Button>
      </div>

      {/* Time Range Selector */}
      <div className="flex gap-2">
        {(['7d', '30d', '90d'] as const).map((range) => (
          <Button
            key={range}
            size="sm"
            onClick={() => setTimeRange(range)}
            className={cn(
              'rounded-xl border transition-all duration-300',
              timeRange === range
                ? 'bg-linear-to-r from-slate-800 to-slate-900 text-white border-transparent shadow-lg'
                : 'bg-slate-50 border-slate-200 text-slate-600 hover:bg-slate-100 hover:text-slate-900'
            )}
          >
            <Calendar className="w-4 h-4 mr-1" />
            {range === '7d' ? 'Last 7 Days' : range === '30d' ? 'Last 30 Days' : 'Last 90 Days'}
          </Button>
        ))}
      </div>

      {/* Main metrics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-linear-to-br from-orange-50 to-red-50 rounded-2xl border border-orange-200 shadow-sm p-6">
          <div className="flex items-start justify-between">
            <div>
              <p className="text-sm text-slate-600">Total Water Loss ({timeRange})</p>
              <p className="text-3xl font-bold text-slate-900 mt-2">28,450 m³</p>
              <p className="text-xs text-red-600 mt-2 flex items-center gap-1">
                <TrendingUp className="w-4 h-4" />
                8.5% above normal
              </p>
            </div>
            <div className="p-3 rounded-xl bg-linear-to-br from-orange-400 to-red-500 shadow-lg">
              <BarChart3 className="w-6 h-6 text-white" />
            </div>
          </div>
        </div>
        <div className="bg-linear-to-br from-blue-50 to-indigo-50 rounded-2xl border border-blue-200 shadow-sm p-6">
          <div className="flex items-start justify-between">
            <div>
              <p className="text-sm text-slate-600">Average Daily Loss ({timeRange})</p>
              <p className="text-3xl font-bold text-slate-900 mt-2">2,190 m³</p>
              <p className="text-xs text-slate-500 mt-2">Per day average</p>
            </div>
            <div className="p-3 rounded-xl bg-linear-to-br from-blue-400 to-indigo-500 shadow-lg">
              <Activity className="w-6 h-6 text-white" />
            </div>
          </div>
        </div>
        <div className="bg-linear-to-br from-amber-50 to-orange-50 rounded-2xl border border-amber-200 shadow-sm p-6">
          <div className="flex items-start justify-between">
            <div>
              <p className="text-sm text-slate-600">Anomalies Detected ({timeRange})</p>
              <p className="text-3xl font-bold text-amber-600 mt-2">127</p>
              <p className="text-xs text-slate-500 mt-2">Across all zones</p>
            </div>
            <div className="p-3 rounded-xl bg-linear-to-br from-amber-400 to-orange-500 shadow-lg">
              <AlertTriangle className="w-6 h-6 text-white" />
            </div>
          </div>
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Water Loss Trend */}
        <div className="bg-white/80 backdrop-blur-sm rounded-2xl border border-slate-200 shadow-lg p-6">
          <h3 className="text-lg font-semibold text-slate-900 mb-6">Water Loss Trend</h3>
          <ResponsiveContainer width="100%" height={350}>
            <AreaChart data={waterLossData}>
              <defs>
                <linearGradient id="colorLossAnalytics" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#f97316" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#f97316" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="colorExpectedAnalytics" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#334155" stopOpacity={0.2} />
                  <stop offset="95%" stopColor="#334155" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.1)" />
              <XAxis dataKey="date" stroke="rgba(0,0,0,0.4)" fontSize={12} />
              <YAxis stroke="rgba(0,0,0,0.4)" fontSize={12} />
              <Tooltip contentStyle={{ backgroundColor: 'white', border: '1px solid rgba(0,0,0,0.1)', borderRadius: '12px', color: '#0f172a', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)' }} />
              <Legend />
              <Area type="monotone" dataKey="loss" stroke="#f97316" strokeWidth={2} fillOpacity={1} fill="url(#colorLossAnalytics)" name="Actual Loss" />
              <Area type="monotone" dataKey="expected" stroke="#334155" strokeWidth={2} fillOpacity={1} fill="url(#colorExpectedAnalytics)" name="Expected Loss" />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Zone Comparison */}
        <div className="bg-white/80 backdrop-blur-sm rounded-2xl border border-slate-200 shadow-lg p-6">
          <h3 className="text-lg font-semibold text-slate-900 mb-6">Water Loss by Zone</h3>
          <ResponsiveContainer width="100%" height={350}>
            <BarChart data={zoneComparisonData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.1)" />
              <XAxis dataKey="name" stroke="rgba(0,0,0,0.4)" fontSize={12} />
              <YAxis stroke="rgba(0,0,0,0.4)" fontSize={12} />
              <Tooltip contentStyle={{ backgroundColor: 'white', border: '1px solid rgba(0,0,0,0.1)', borderRadius: '12px', color: '#0f172a', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)' }} />
              <Bar dataKey="waterLoss" fill="#f97316" radius={[8, 8, 0, 0]} name="Water Loss (m³)" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Pressure vs Flow */}
        <div className="bg-white/80 backdrop-blur-sm rounded-2xl border border-slate-200 shadow-lg p-6">
          <h3 className="text-lg font-semibold text-slate-900 mb-6">Pressure vs Flow Rate by Zone</h3>
          <ResponsiveContainer width="100%" height={350}>
            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.1)" />
              <XAxis dataKey="pressure" name="Pressure (bar)" stroke="rgba(0,0,0,0.4)" fontSize={12} />
              <YAxis dataKey="flowRate" name="Flow Rate (L/min)" stroke="rgba(0,0,0,0.4)" fontSize={12} />
              <Tooltip contentStyle={{ backgroundColor: 'white', border: '1px solid rgba(0,0,0,0.1)', borderRadius: '12px', color: '#0f172a', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)' }} />
              <Legend />
              <Scatter name="Zones" data={zoneComparisonData} fill="#334155" />
            </ScatterChart>
          </ResponsiveContainer>
        </div>

        {/* Anomaly Detection */}
        <div className="bg-white/80 backdrop-blur-sm rounded-2xl border border-slate-200 shadow-lg p-6">
          <h3 className="text-lg font-semibold text-slate-900 mb-6">Anomaly Detection Pattern</h3>
          <ResponsiveContainer width="100%" height={350}>
            <ScatterChart>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.1)" />
              <XAxis dataKey="x" type="number" stroke="rgba(0,0,0,0.4)" fontSize={12} />
              <YAxis dataKey="y" stroke="rgba(0,0,0,0.4)" fontSize={12} />
              <Tooltip contentStyle={{ backgroundColor: 'white', border: '1px solid rgba(0,0,0,0.1)', borderRadius: '12px', color: '#0f172a', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)' }} />
              <Legend />
              <Scatter name="Normal" data={anomalyData.filter((d) => d.type === 'Normal')} fill="#10b981" />
              <Scatter name="Anomaly" data={anomalyData.filter((d) => d.type === 'Anomaly')} fill="#f97316" />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Insights */}
      <div className="bg-white/80 backdrop-blur-sm rounded-2xl border border-slate-200 shadow-lg p-6">
        <h3 className="text-lg font-semibold text-slate-900 mb-6">Key Insights</h3>
        <div className="space-y-3">
          <div className="flex gap-3 p-4 bg-linear-to-r from-amber-50 to-amber-100/50 rounded-xl border border-amber-200">
            <div className="w-2 h-2 bg-amber-500 rounded-full mt-1.5 shrink-0 animate-pulse"></div>
            <p className="text-sm text-slate-700">
              <span className="font-medium text-slate-900">Zone A</span> shows 8.5% higher water loss than expected. Recommend immediate inspection of main pipeline sector 2.
            </p>
          </div>
          <div className="flex gap-3 p-4 bg-linear-to-r from-blue-50 to-blue-100/50 rounded-xl border border-blue-200">
            <div className="w-2 h-2 bg-blue-500 rounded-full mt-1.5 shrink-0 animate-pulse"></div>
            <p className="text-sm text-slate-700">
              <span className="font-medium text-slate-900">127 anomalies</span> detected in the past {timeRange}. Most anomalies occur during peak demand hours (9 AM - 12 PM).
            </p>
          </div>
          <div className="flex gap-3 p-4 bg-linear-to-r from-emerald-50 to-emerald-100/50 rounded-xl border border-emerald-200">
            <div className="w-2 h-2 bg-emerald-500 rounded-full mt-1.5 shrink-0 animate-pulse"></div>
            <p className="text-sm text-slate-700">
              <span className="font-medium text-slate-900">Zone C</span> is performing optimally with 12% lower loss than average. Best practices from this zone should be replicated.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
