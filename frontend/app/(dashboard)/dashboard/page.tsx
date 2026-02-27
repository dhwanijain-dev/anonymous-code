'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { AlertCircle, TrendingUp, Droplet, Zap, Activity, Waves, AlertTriangle, RefreshCw, Radio, Clock } from 'lucide-react';
import { DashboardMetrics, Alert } from '@/lib/types';
import { KPICard } from '@/components/dashboard/kpi-card';
import { AlertFeed } from '@/components/dashboard/alert-feed';
import { MiniMap } from '@/components/dashboard/mini-map';
import { PressureHeatmap } from '@/components/dashboard/pressure-heatmap';
import { PipelineNetworkGraph } from '@/components/dashboard/pipeline-network-graph';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Eye, Brain } from 'lucide-react';
import { backendService, DerivedMetrics, DerivedAlert } from '@/lib/backend-service';

// Types for BIWS LSTM burst prediction
interface BurstNode {
  node_id: string;
  time_to_burst: number;
  risk_score: number;
  urgency: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW';
}

interface PredictionInfo {
  top_nodes: BurstNode[];
  n_nodes_scored: number;
  horizon_min: number;
  buffer_ready: boolean;
  model_loaded: boolean;
}

interface PressureHistoryPoint {
  time: string;
  [key: string]: string | number;
}

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

// Default metrics (used before data loads)
const defaultMetrics: DashboardMetrics = {
  total_water_loss: 0,
  total_water_loss_percentage: 0,
  critical_alerts: 0,
  active_leaks: 0,
  sensor_health_percentage: 100,
  zones_with_issues: 0,
  avg_water_loss_24h: 0,
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

export default function DashboardPage() {
  const [isLoading, setIsLoading] = useState(true);
  const [pipelineData, setPipelineData] = useState<PipelineData | null>(null);
  const [pressureChartData, setPressureChartData] = useState<{ node: string; pressure: number }[]>([]);
  const [networkData, setNetworkData] = useState<NetworkData | null>(null);
  const [metrics, setMetrics] = useState<DashboardMetrics>(defaultMetrics);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [anomalyData, setAnomalyData] = useState([
    { name: 'Normal', value: 85, color: '#06b6d4' },
    { name: 'Anomaly', value: 10, color: '#f97316' },
    { name: 'Critical', value: 5, color: '#ef4444' },
  ]);
  const [modelMetrics, setModelMetrics] = useState<{accuracy?: number; precision?: number; recall?: number; f1?: number}>({});

  // BIWS Dashboard State
  const [pressureHistory, setPressureHistory] = useState<PressureHistoryPoint[]>([]);
  const [trackedNodes, setTrackedNodes] = useState<string[]>([]);
  const [burstPrediction, setBurstPrediction] = useState<PredictionInfo | null>(null);
  const [isNetworkAnomaly, setIsNetworkAnomaly] = useState(false);
  const [networkLeakCount, setNetworkLeakCount] = useState(0);
  const [totalMonitoredNodes, setTotalMonitoredNodes] = useState(0);
  const trackedNodesRef = useRef<string[]>([]);

  // Fetch all dashboard data from backend
  const fetchDashboardData = useCallback(async () => {
    try {
      // Fetch metrics from backend
      const backendMetrics = await backendService.getMetrics();
      setMetrics({
        total_water_loss: backendMetrics.total_water_loss,
        total_water_loss_percentage: backendMetrics.total_water_loss_percentage,
        critical_alerts: backendMetrics.critical_alerts,
        active_leaks: backendMetrics.active_leaks,
        sensor_health_percentage: backendMetrics.sensor_health_percentage,
        zones_with_issues: backendMetrics.zones_with_issues,
        avg_water_loss_24h: backendMetrics.avg_water_loss_24h,
      });

      // Store model metrics
      setModelMetrics({
        accuracy: backendMetrics.model_accuracy,
        precision: backendMetrics.model_precision,
        recall: backendMetrics.model_recall,
        f1: backendMetrics.model_f1,
      });

      // Fetch alerts from backend
      const backendAlerts = await backendService.getAlerts();
      setAlerts(backendAlerts.map(a => ({
        id: a.id,
        zone_id: a.zone_id,
        sensor_id: a.sensor_id,
        alert_type: a.alert_type,
        severity: a.severity,
        status: a.status,
        description: a.description,
        location: a.location,
        estimated_loss: a.estimated_loss,
        created_at: a.created_at,
        acknowledged_at: a.acknowledged_at,
        resolved_at: a.resolved_at,
      })));

      // Calculate anomaly distribution based on current data
      const snapshot = backendService.getLatestSnapshot();
      if (snapshot) {
        const totalNodes = snapshot.nodes.length;
        const criticalCount = snapshot.nodes.filter(n => n.is_leaking === 1).length;
        const anomalyCount = snapshot.nodes.filter(n => !n.is_leaking && (n.pred_leak_prob || 0) > 0.3).length;
        const normalCount = totalNodes - criticalCount - anomalyCount;

        setAnomalyData([
          { name: 'Normal', value: Math.round((normalCount / totalNodes) * 100), color: '#06b6d4' },
          { name: 'Anomaly', value: Math.round((anomalyCount / totalNodes) * 100), color: '#f97316' },
          { name: 'Critical', value: Math.round((criticalCount / totalNodes) * 100), color: '#ef4444' },
        ]);
      }

    } catch (error) {
      console.error('Failed to fetch dashboard data from backend:', error);
    }
  }, []);

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

        // ── BIWS Dashboard Data Extraction ──
        // Extract data from whichever format we got
        const db = json.data?.[0] || json;
        if (db && db.nodes) {
          const now = new Date().toLocaleTimeString();
          setTotalMonitoredNodes(Array.isArray(db.nodes) ? db.nodes.length : 0);

          // Network anomaly status
          if (db.metrics?.network) {
            const net = db.metrics.network;
            const isLeaking = net.y_true?.[0] || net.tp > 0 || net.fp > 0;
            setIsNetworkAnomaly(!!isLeaking);
            setNetworkLeakCount(isLeaking ? (net.n || 1) : 0);
          }

          // Pick 3 tracked nodes on first run, then keep them stable
          if (trackedNodesRef.current.length === 0 && Array.isArray(db.nodes) && db.nodes.length > 0) {
            const shuffled = [...db.nodes].sort(() => 0.5 - Math.random());
            const picked = shuffled.slice(0, 3).map((n: GraphNode) => n.id);
            trackedNodesRef.current = picked;
            setTrackedNodes(picked);
          }

          // Build pressure history point
          if (trackedNodesRef.current.length > 0 && Array.isArray(db.nodes)) {
            const point: PressureHistoryPoint = { time: now };
            trackedNodesRef.current.forEach((nodeId: string) => {
              const found = db.nodes.find((n: GraphNode) => n.id === nodeId);
              point[`Node ${nodeId}`] = found ? found.pressure : 0;
            });
            setPressureHistory(prev => {
              const updated = [...prev, point];
              return updated.length > 20 ? updated.slice(-20) : updated;
            });
          }

          // LSTM burst predictions
          if (db.prediction) {
            setBurstPrediction(db.prediction);
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
    fetchDashboardData(); // Also fetch metrics and alerts
    const interval = setInterval(() => {
      fetchPipelineData();
      fetchDashboardData();
    }, 5000);
    
    return () => clearInterval(interval);
  }, [fetchDashboardData]);

  useEffect(() => {
    // Initial load complete
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
          value={`${metrics.total_water_loss.toLocaleString()} m³`}
          change={metrics.total_water_loss_percentage}
          unit="%"
          icon={Droplet}
          trend={metrics.total_water_loss_percentage > 10 ? "up" : "down"}
          status={metrics.total_water_loss_percentage > 15 ? "critical" : metrics.total_water_loss_percentage > 10 ? "warning" : "success"}
        />
        <KPICard
          title="Critical Alerts"
          value={metrics.critical_alerts}
          change={metrics.critical_alerts}
          unit="active"
          icon={AlertCircle}
          trend={metrics.critical_alerts > 0 ? "up" : "down"}
          status={metrics.critical_alerts > 5 ? "critical" : metrics.critical_alerts > 0 ? "warning" : "success"}
        />
        <KPICard
          title="Active Leaks"
          value={metrics.active_leaks}
          change={metrics.active_leaks}
          unit="sites"
          icon={Zap}
          trend={metrics.active_leaks > 0 ? "up" : "down"}
          status={metrics.active_leaks > 3 ? "critical" : metrics.active_leaks > 0 ? "warning" : "success"}
        />
        <KPICard
          title="Sensor Health"
          value={`${metrics.sensor_health_percentage}%`}
          change={100 - metrics.sensor_health_percentage}
          unit="operational"
          icon={TrendingUp}
          trend={metrics.sensor_health_percentage >= 90 ? "down" : "up"}
          status={metrics.sensor_health_percentage >= 90 ? "success" : metrics.sensor_health_percentage >= 75 ? "warning" : "critical"}
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

      {/* ── BIWS Detection & Time-to-Burst Dashboard ── */}
      <div className="bg-white/80 backdrop-blur-sm rounded-2xl border border-slate-200 shadow-lg p-6">
        {/* Card Header */}
        <div className="text-center mb-6">
          <h2 className="text-2xl font-bold text-slate-900">
            Global Detection &amp; Time-to-Burst Dashboard
          </h2>
          <p className="text-slate-500 text-sm mt-1">
            Real-time BIWS network simulation and predictive intelligence
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left: Network Status Monitor (2/3 width) */}
          <div className="lg:col-span-2 bg-white/80 backdrop-blur-sm rounded-2xl border border-slate-200 shadow-lg p-6">
            <div className="flex items-center justify-between mb-5">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-xl bg-gradient-to-br from-blue-500 to-cyan-500 shadow-lg">
                  <Radio className="w-5 h-5 text-white" />
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-slate-900">Network Status Monitor</h3>
                  <p className="text-sm text-slate-500">Live pressure tracking</p>
                </div>
              </div>
              <span className={`text-xs font-semibold px-3 py-1.5 rounded-full ${
                isNetworkAnomaly
                  ? 'bg-red-100 text-red-700'
                  : 'bg-emerald-100 text-emerald-700'
              }`}>
                {isNetworkAnomaly ? 'NETWORK ANOMALY DETECTED' : 'SYSTEM NORMAL'}
              </span>
            </div>

            {/* Live Pressure Chart */}
            <div className="h-[300px]">
              {pressureHistory.length > 1 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={pressureHistory}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.1)" />
                    <XAxis dataKey="time" stroke="rgba(0,0,0,0.4)" fontSize={11} tick={{ fill: '#64748b' }} />
                    <YAxis stroke="rgba(0,0,0,0.4)" fontSize={11} tick={{ fill: '#64748b' }} label={{ value: 'Pressure', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 12 }} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: 'white',
                        border: '1px solid rgba(0,0,0,0.1)',
                        borderRadius: '12px',
                        color: '#0f172a',
                        boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
                      }}
                    />
                    {trackedNodes.map((nodeId, i) => (
                      <Line
                        key={nodeId}
                        type="monotone"
                        dataKey={`Node ${nodeId}`}
                        stroke={['#3b82f6', '#f97316', '#06b6d4'][i] || '#3b82f6'}
                        strokeWidth={2}
                        dot={false}
                        isAnimationActive={false}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div className="h-full flex items-center justify-center">
                  <div className="text-center">
                    <div className="w-10 h-10 border-2 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-3" />
                    <p className="text-sm text-slate-500">Waiting for pressure data...</p>
                  </div>
                </div>
              )}
            </div>

            {/* Metrics Grid */}
            <div className="grid grid-cols-2 gap-4 mt-5">
              <div className="bg-slate-50 rounded-xl p-4 text-center border border-slate-200">
                <div className="text-3xl font-bold text-blue-600">{totalMonitoredNodes}</div>
                <div className="text-xs text-slate-500 mt-1 uppercase tracking-wider">Nodes Monitored</div>
              </div>
              <div className="bg-slate-50 rounded-xl p-4 text-center border border-slate-200">
                <div className="text-3xl font-bold text-red-600">{networkLeakCount}</div>
                <div className="text-xs text-slate-500 mt-1 uppercase tracking-wider">Network Leaks Detected</div>
              </div>
            </div>
          </div>

          {/* Right: LSTM Burst Predictions (1/3 width) */}
          <div className="bg-white/80 backdrop-blur-sm rounded-2xl border border-slate-200 shadow-lg p-6">
            <div className="flex items-center justify-between mb-5">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-xl bg-gradient-to-br from-purple-500 to-pink-500 shadow-lg">
                  <Clock className="w-5 h-5 text-white" />
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-slate-900">LSTM Burst Predictions</h3>
                  <p className="text-sm text-slate-500">Time-to-burst forecast</p>
                </div>
              </div>
              <span className={`text-xs font-semibold px-3 py-1.5 rounded-full ${
                burstPrediction?.buffer_ready
                  ? 'bg-emerald-100 text-emerald-700'
                  : 'bg-amber-100 text-amber-700'
              }`}>
                {burstPrediction?.buffer_ready ? 'LIVE PREDICTION' : 'BUILDING BUFFER...'}
              </span>
            </div>

            {/* Node Cards */}
            <div className="flex flex-col gap-3 max-h-[460px] overflow-y-auto pr-2">
              {burstPrediction?.top_nodes && burstPrediction.top_nodes.length > 0 ? (
                burstPrediction.top_nodes.map((node) => (
                  <div
                    key={node.node_id}
                    className={`rounded-xl p-4 flex items-center justify-between border transition-all duration-300 ${
                      node.urgency === 'CRITICAL' ? 'bg-red-50 border-red-200' :
                      node.urgency === 'HIGH' ? 'bg-amber-50 border-amber-200' :
                      node.urgency === 'MEDIUM' ? 'bg-blue-50 border-blue-200' :
                      'bg-slate-50 border-slate-200'
                    }`}
                  >
                    <div>
                      <div className="font-semibold text-slate-900 text-base">{node.node_id}</div>
                      <div className="text-xs text-slate-500 mt-1">
                        Risk: {node.risk_score.toFixed(2)}
                      </div>
                    </div>
                    <div className="text-right">
                      <div className={`text-xl font-bold ${
                        node.urgency === 'CRITICAL' ? 'text-red-600' :
                        node.urgency === 'HIGH' ? 'text-amber-600' :
                        node.urgency === 'MEDIUM' ? 'text-blue-600' :
                        'text-slate-600'
                      }`}>
                        {node.time_to_burst.toFixed(1)}{' '}
                        <span className="text-sm font-normal text-slate-500">min</span>
                      </div>
                      <div className={`text-xs font-semibold uppercase ${
                        node.urgency === 'CRITICAL' ? 'text-red-600' :
                        node.urgency === 'HIGH' ? 'text-amber-600' :
                        node.urgency === 'MEDIUM' ? 'text-blue-600' :
                        'text-slate-500'
                      }`}>{node.urgency}</div>
                    </div>
                  </div>
                ))
              ) : (
                <div className="h-48 flex items-center justify-center">
                  <div className="text-center">
                    <div className="w-10 h-10 border-2 border-purple-500 border-t-transparent rounded-full animate-spin mx-auto mb-3" />
                    <p className="text-sm text-slate-500">Waiting for LSTM predictions...</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
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
                  data={anomalyData}
                  cx="50%"
                  cy="50%"
                  innerRadius={50}
                  outerRadius={80}
                  paddingAngle={3}
                  dataKey="value"
                >
                  {anomalyData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="space-y-2 mt-4">
            {anomalyData.map((item) => (
              <div key={item.name} className="flex items-center justify-between text-sm">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full" style={{ backgroundColor: item.color }} />
                  <span className="text-slate-600">{item.name}</span>
                </div>
                <span className="text-slate-900 font-medium">{item.value}%</span>
              </div>
            ))}
          </div>
          {/* Model Metrics */}
          {modelMetrics.accuracy !== undefined && (
            <div className="mt-4 pt-4 border-t border-slate-200">
              <p className="text-xs text-slate-500 mb-2">Model Performance</p>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="flex justify-between">
                  <span className="text-slate-500">Accuracy:</span>
                  <span className="font-medium text-slate-700">{(modelMetrics.accuracy! * 100).toFixed(1)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Precision:</span>
                  <span className="font-medium text-slate-700">{modelMetrics.precision ? (modelMetrics.precision * 100).toFixed(1) : 'N/A'}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Recall:</span>
                  <span className="font-medium text-slate-700">{modelMetrics.recall ? (modelMetrics.recall * 100).toFixed(1) : 'N/A'}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">F1 Score:</span>
                  <span className="font-medium text-slate-700">{modelMetrics.f1 ? (modelMetrics.f1 * 100).toFixed(1) : 'N/A'}%</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Mini Map and Alert Feed */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <MiniMap />
        <div className="lg:col-span-2">
          <AlertFeed alerts={alerts} />
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
    