'use client';

import { useState, useEffect } from 'react';
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { AlertCircle, TrendingUp, Droplet, Zap, Activity, Waves, AlertTriangle } from 'lucide-react';
import { DashboardMetrics, Alert } from '@/lib/types';
import { KPICard } from '@/components/dashboard/kpi-card';
import { AlertFeed } from '@/components/dashboard/alert-feed';
import { MiniMap } from '@/components/dashboard/mini-map';
import { PressureHeatmap } from '@/components/dashboard/pressure-heatmap';
import { PipelineNetworkGraph } from '@/components/dashboard/pipeline-network-graph';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Eye, Brain } from 'lucide-react';

// Types for pipeline data
interface RiskNode {
  node: string;
  risk_level: string;
  pressure: number;
}

interface PipelineData {
  nodes: string[];
  pressure: number[];
  risk_nodes: RiskNode[];
  timestamp?: number;
}

// Types for network graph data (from /stream endpoint)
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

interface NetworkData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

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
  const [pipelineData, setPipelineData] = useState<PipelineData | null>(null);
  const [pressureChartData, setPressureChartData] = useState<{ node: string; pressure: number }[]>([]);
  const [networkData, setNetworkData] = useState<NetworkData | null>(null);

  // Fetch pipeline pressure data from backend API
  useEffect(() => {
    const fetchPipelineData = async () => {
      try {
        const res = await fetch('http://127.0.0.1:8000/stream');
        const json = await res.json();
        
        // Check for new data format with nodes/edges objects
        if (json.nodes && json.edges) {
          // New format: nodes and edges are objects with full data
          setNetworkData({
            nodes: json.nodes,
            edges: json.edges,
          });
          
          // Also create pressure chart data from nodes
          const chartData = json.nodes.slice(0, 20).map((node: GraphNode) => ({
            node: node.id,
            pressure: node.pressure,
          }));
          setPressureChartData(chartData);
          
          // Create risk nodes from leaking nodes
          const riskNodes: RiskNode[] = json.nodes
            .filter((n: GraphNode) => n.is_leaking === 1 || (n.pred_leak_prob && n.pred_leak_prob > 0.3))
            .map((n: GraphNode) => ({
              node: n.id,
              risk_level: n.is_leaking === 1 ? 'high' : n.pred_leak_prob && n.pred_leak_prob > 0.5 ? 'medium' : 'low',
              pressure: n.pressure,
            }));
          
          setPipelineData({
            nodes: json.nodes.map((n: GraphNode) => n.id),
            pressure: json.nodes.map((n: GraphNode) => n.pressure),
            risk_nodes: riskNodes,
            timestamp: json.timestamp,
          });
        } else if (json.data && json.data.length > 0) {
          // Data array format from /stream endpoint
          const latest = json.data[0];
          
          // Set network data for the PipelineNetworkGraph component
          if (latest.nodes && latest.edges) {
            setNetworkData({
              nodes: latest.nodes,
              edges: latest.edges,
            });
            
            // Create pressure chart data from nodes
            const chartData = latest.nodes.slice(0, 20).map((node: GraphNode) => ({
              node: node.id,
              pressure: node.pressure,
            }));
            setPressureChartData(chartData);
            
            // Create risk nodes from leaking nodes
            const riskNodes: RiskNode[] = latest.nodes
              .filter((n: GraphNode) => n.is_leaking === 1 || (n.pred_leak_prob && n.pred_leak_prob > 0.3))
              .map((n: GraphNode) => ({
                node: n.id,
                risk_level: n.is_leaking === 1 ? 'high' : n.pred_leak_prob && n.pred_leak_prob > 0.5 ? 'medium' : 'low',
                pressure: n.pressure,
              }));
            
            setPipelineData({
              nodes: latest.nodes.map((n: GraphNode) => n.id),
              pressure: latest.nodes.map((n: GraphNode) => n.pressure),
              risk_nodes: riskNodes,
              timestamp: latest.timestamp,
            });
          } else {
            // Legacy format with nodes as string array
            const chartData = latest.nodes.map((node: string, index: number) => ({
              node,
              pressure: latest.pressure[index],
            }));
            setPressureChartData(chartData);
            
            // Transform risk_nodes from array of strings to array of objects
            const transformedRiskNodes: RiskNode[] = (latest.risk_nodes || []).map((nodeName: string) => {
              const nodeIndex = latest.nodes.indexOf(nodeName);
              const pressure = nodeIndex >= 0 ? latest.pressure[nodeIndex] : 0;
              const risk_level = pressure < 0.5 ? 'high' : pressure < 1.0 ? 'medium' : 'low';
              return { node: nodeName, risk_level, pressure };
            });
            
            setPipelineData({
              nodes: latest.nodes,
              pressure: latest.pressure,
              risk_nodes: transformedRiskNodes,
              timestamp: latest.timestamp,
            });
          }
        }
      } catch (error) {
        console.log('API fetch error, using mock data:', error);
        // Use mock data if API fails
        const mockPipelineData: PipelineData = {
          nodes: ['Node A', 'Node B', 'Node C', 'Node D', 'Node E', 'Node F'],
          pressure: [3.2, 3.5, 2.8, 3.1, 2.5, 3.8],
          risk_nodes: [
            { node: 'Node C', risk_level: 'medium', pressure: 2.8 },
            { node: 'Node E', risk_level: 'high', pressure: 2.5 },
          ],
        };
        setPipelineData(mockPipelineData);
        setPressureChartData(
          mockPipelineData.nodes.map((node, index) => ({
            node,
            pressure: mockPipelineData.pressure[index],
          }))
        );
      }
    };

    fetchPipelineData();
    const interval = setInterval(fetchPipelineData, 5000);
    
    return () => clearInterval(interval);
  }, []);

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

      {/* Pressure Heatmap - Full viewport height */}
      <PressureHeatmap 
        data={pipelineData ? {
          nodes: pipelineData.nodes,
          pressure: pipelineData.pressure,
          riskNodes: pipelineData.risk_nodes.map(r => r.node)
        } : null}
        isLoading={!pipelineData}
      />

      {/* Pipeline Network Graph with Tabs */}
      <div className="bg-white/80 backdrop-blur-sm rounded-2xl border border-slate-200 shadow-lg p-6">
        <Tabs defaultValue="detection" className="w-full">
          <TabsList className="grid w-full max-w-md grid-cols-2 mb-6">
            <TabsTrigger value="detection" className="flex items-center gap-2">
              <Eye className="w-4 h-4" />
              Detection System
            </TabsTrigger>
            <TabsTrigger value="predictive" className="flex items-center gap-2">
              <Brain className="w-4 h-4" />
              Predictive System
            </TabsTrigger>
          </TabsList>
          
          <TabsContent value="detection">
            <div className="mb-4">
              <h3 className="text-lg font-semibold text-slate-900">Real-time Leak Detection</h3>
              <p className="text-sm text-slate-500">Monitoring active leaks in the pipeline network</p>
            </div>
            <PipelineNetworkGraph
              nodes={networkData?.nodes || []}
              edges={networkData?.edges || []}
              isLoading={!networkData}
              mode="detection"
            />
          </TabsContent>
          
          <TabsContent value="predictive">
            <div className="mb-4">
              <h3 className="text-lg font-semibold text-slate-900">Predictive Leak Analysis</h3>
              <p className="text-sm text-slate-500">ML-based prediction of potential leak locations</p>
            </div>
            <PipelineNetworkGraph
              nodes={networkData?.nodes || []}
              edges={networkData?.edges || []}
              isLoading={!networkData}
              mode="predictive"
            />
          </TabsContent>
        </Tabs>
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

      {/* Pipeline Pressure Monitoring */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Pressure Stream Chart */}
        <div className="lg:col-span-2 bg-white/80 backdrop-blur-sm rounded-2xl border border-slate-200 shadow-lg p-6">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 rounded-xl bg-gradient-to-br from-blue-500 to-cyan-500 shadow-lg">
              <Waves className="w-5 h-5 text-white" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-slate-900">Pipeline Pressure Monitoring</h3>
              <p className="text-sm text-slate-500">Real-time pressure stream data</p>
            </div>
          </div>
          
          {pressureChartData.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={pressureChartData}>
                <defs>
                  <linearGradient id="colorPressure" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#0ea5e9" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#0ea5e9" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.1)" />
                <XAxis dataKey="node" stroke="rgba(0,0,0,0.4)" fontSize={12} />
                <YAxis stroke="rgba(0,0,0,0.4)" fontSize={12} unit=" bar" />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: 'white', 
                    border: '1px solid rgba(0,0,0,0.1)',
                    borderRadius: '12px',
                    color: '#0f172a',
                    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                  }}
                  formatter={(value: number) => [`${value.toFixed(2)} bar`, 'Pressure']}
                />
                <Line 
                  type="monotone" 
                  dataKey="pressure" 
                  stroke="#0ea5e9" 
                  strokeWidth={3}
                  dot={{ fill: '#0ea5e9', strokeWidth: 2, r: 5 }}
                  activeDot={{ r: 8, fill: '#0284c7' }}
                />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-[300px] flex items-center justify-center">
              <div className="text-center">
                <div className="w-10 h-10 border-3 border-slate-600 border-t-transparent rounded-full animate-spin mx-auto mb-3" />
                <p className="text-sm text-slate-500">Loading pressure data...</p>
              </div>
            </div>
          )}
        </div>

        {/* Risk Nodes Panel */}
        <div className="bg-white/80 backdrop-blur-sm rounded-2xl border h-120 overflow-y-scroll border-slate-200 shadow-lg p-6">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 rounded-xl bg-gradient-to-br from-amber-500 to-orange-500 shadow-lg">
              <AlertTriangle className="w-5 h-5 text-white" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-slate-900">Risk Nodes</h3>
              <p className="text-sm text-slate-500">Nodes requiring attention</p>
            </div>
          </div>

          {pipelineData?.risk_nodes && pipelineData.risk_nodes.length > 0 ? (
            <div className="space-y-3">
              {pipelineData.risk_nodes.map((riskNode, index) => (
                <div 
                  key={index}
                  className={`p-4 rounded-xl border transition-all duration-300 ${
                    riskNode.risk_level === 'high' 
                      ? 'bg-red-50 border-red-200' 
                      : riskNode.risk_level === 'medium'
                      ? 'bg-amber-50 border-amber-200'
                      : 'bg-blue-50 border-blue-200'
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium text-slate-900">{riskNode.node}</span>
                    <span className={`text-xs font-semibold px-2 py-1 rounded-full ${
                      riskNode.risk_level === 'high'
                        ? 'bg-red-100 text-red-700'
                        : riskNode.risk_level === 'medium'
                        ? 'bg-amber-100 text-amber-700'
                        : 'bg-blue-100 text-blue-700'
                    }`}>
                      {riskNode.risk_level.toUpperCase()}
                    </span>
                  </div>
                  <div className="text-sm text-slate-600">
                    Pressure: <span className="font-medium text-slate-900">{riskNode.pressure} bar</span>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="h-48 flex items-center justify-center">
              <div className="text-center">
                <div className="w-12 h-12 mx-auto mb-3 rounded-full bg-emerald-100 flex items-center justify-center">
                  <Activity className="w-6 h-6 text-emerald-600" />
                </div>
                <p className="text-sm text-slate-600 font-medium">All nodes operating normally</p>
                <p className="text-xs text-slate-400 mt-1">No risk nodes detected</p>
              </div>
            </div>
          )}

          {/* Raw Data Preview */}
          {pipelineData && (
            <div className="mt-4 pt-4 border-t border-slate-200">
              <p className="text-xs text-slate-500 mb-2">Raw Data Preview</p>
              <pre className="text-xs bg-slate-50 rounded-lg p-3 overflow-auto max-h-32 text-slate-700">
                {JSON.stringify(pipelineData.risk_nodes, null, 2)}
              </pre>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
