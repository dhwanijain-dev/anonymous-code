'use client';

import { useState, useEffect, useCallback } from 'react';
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
import { TrendingUp, Calendar, Download, BarChart3, Activity, AlertTriangle, RefreshCw } from 'lucide-react';
import { cn } from '@/lib/utils';
import { backendService } from '@/lib/backend-service';

export default function AnalyticsPage() {
  const [timeRange, setTimeRange] = useState<'7d' | '30d' | '90d'>('30d');
  const [isLoading, setIsLoading] = useState(true);
  const [analyticsData, setAnalyticsData] = useState<{
    waterLossData: { date: string; loss: number; expected: number }[];
    zoneData: { name: string; waterLoss: number; pressure: number; flowRate: number }[];
    anomalyData: { x: number; y: number; type: string }[];
  }>({
    waterLossData: [],
    zoneData: [],
    anomalyData: [],
  });
  const [metrics, setMetrics] = useState({
    totalWaterLoss: 0,
    avgDailyLoss: 0,
    anomaliesDetected: 0,
    percentAboveNormal: 0,
  });

  // Fetch analytics data from backend
  const fetchAnalytics = useCallback(async () => {
    try {
      setIsLoading(true);
      const data = await backendService.getAnalyticsData();
      setAnalyticsData(data);

      // Calculate metrics
      const backendMetrics = await backendService.getMetrics();
      const totalLoss = data.waterLossData.reduce((sum, d) => sum + d.loss, 0);
      const totalExpected = data.waterLossData.reduce((sum, d) => sum + d.expected, 0);
      const percentAbove = totalExpected > 0 ? ((totalLoss - totalExpected) / totalExpected) * 100 : 0;

      setMetrics({
        totalWaterLoss: totalLoss,
        avgDailyLoss: Math.round(totalLoss / Math.max(data.waterLossData.length, 1)),
        anomaliesDetected: backendMetrics.critical_alerts + backendMetrics.active_leaks,
        percentAboveNormal: Math.round(percentAbove * 10) / 10,
      });

      setIsLoading(false);
    } catch (error) {
      console.error('Failed to fetch analytics:', error);
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchAnalytics();
    const interval = setInterval(fetchAnalytics, 10000); // Refresh every 10 seconds
    return () => clearInterval(interval);
  }, [fetchAnalytics]);

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-slate-900">Analytics</h1>
          <p className="text-slate-500 mt-1">Water loss trends and system analysis (Live data)</p>
        </div>
        <div className="flex items-center gap-3">
          <Button
            onClick={fetchAnalytics}
            variant="outline"
            size="sm"
            className="flex items-center gap-2"
          >
            <RefreshCw className={cn("w-4 h-4", isLoading && "animate-spin")} />
            Refresh
          </Button>
          <Button className="bg-slate-100 border border-slate-200 text-slate-700 hover:bg-slate-200 rounded-xl shadow-sm">
            <Download className="w-4 h-4 mr-2" />
            Export Report
          </Button>
        </div>
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
              <p className="text-3xl font-bold text-slate-900 mt-2">{metrics.totalWaterLoss.toLocaleString()} m³</p>
              <p className={cn(
                "text-xs mt-2 flex items-center gap-1",
                metrics.percentAboveNormal > 0 ? "text-red-600" : "text-emerald-600"
              )}>
                <TrendingUp className="w-4 h-4" />
                {Math.abs(metrics.percentAboveNormal)}% {metrics.percentAboveNormal > 0 ? 'above' : 'below'} normal
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
              <p className="text-3xl font-bold text-slate-900 mt-2">{metrics.avgDailyLoss.toLocaleString()} m³</p>
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
              <p className="text-3xl font-bold text-amber-600 mt-2">{metrics.anomaliesDetected}</p>
              <p className="text-xs text-slate-500 mt-2">Active leaks & alerts</p>
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
            <AreaChart data={analyticsData.waterLossData}>
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
            <BarChart data={analyticsData.zoneData}>
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
              <Scatter name="Zones" data={analyticsData.zoneData} fill="#334155" />
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
              <Scatter name="Normal" data={analyticsData.anomalyData.filter((d) => d.type === 'Normal')} fill="#10b981" />
              <Scatter name="Anomaly" data={analyticsData.anomalyData.filter((d) => d.type === 'Anomaly')} fill="#f97316" />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Insights */}
      <div className="bg-white/80 backdrop-blur-sm rounded-2xl border border-slate-200 shadow-lg p-6">
        <h3 className="text-lg font-semibold text-slate-900 mb-6">Key Insights (Live from ML Model)</h3>
        <div className="space-y-3">
          {metrics.percentAboveNormal > 5 && (
            <div className="flex gap-3 p-4 bg-linear-to-r from-amber-50 to-amber-100/50 rounded-xl border border-amber-200">
              <div className="w-2 h-2 bg-amber-500 rounded-full mt-1.5 shrink-0 animate-pulse"></div>
              <p className="text-sm text-slate-700">
                <span className="font-medium text-slate-900">Water loss is {metrics.percentAboveNormal.toFixed(1)}% above normal.</span> The ML model has detected elevated leak risk across the network. Recommend immediate inspection.
              </p>
            </div>
          )}
          <div className="flex gap-3 p-4 bg-linear-to-r from-blue-50 to-blue-100/50 rounded-xl border border-blue-200">
            <div className="w-2 h-2 bg-blue-500 rounded-full mt-1.5 shrink-0 animate-pulse"></div>
            <p className="text-sm text-slate-700">
              <span className="font-medium text-slate-900">{metrics.anomaliesDetected} active issues</span> detected by the leak detection model. The system is continuously monitoring all {analyticsData.zoneData.length} zones.
            </p>
          </div>
          <div className="flex gap-3 p-4 bg-linear-to-r from-emerald-50 to-emerald-100/50 rounded-xl border border-emerald-200">
            <div className="w-2 h-2 bg-emerald-500 rounded-full mt-1.5 shrink-0 animate-pulse"></div>
            <p className="text-sm text-slate-700">
              <span className="font-medium text-slate-900">Average daily loss: {metrics.avgDailyLoss.toLocaleString()} m³.</span> The prediction model is actively forecasting potential leak locations.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
