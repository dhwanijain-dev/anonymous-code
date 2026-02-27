/**
 * Backend Service - Integrates with the Python FastAPI WNTR simulation backend
 * Backend runs on http://127.0.0.1:8000
 */

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://127.0.0.1:8000';

// Types matching the backend's data structures
export interface GraphNode {
  id: string;
  x: number;
  y: number;
  pressure: number;
  is_leaking: number;
  pred_leak_prob?: number;
  pred_is_leaking?: number;
}

export interface GraphEdge {
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

export interface BinaryMetrics {
  n: number;
  tp?: number;
  tn?: number;
  fp?: number;
  fn?: number;
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1?: number;
}

export interface StreamMetrics {
  nodes: BinaryMetrics;
  pipes: BinaryMetrics;
  model_loaded: boolean;
}

export interface PredictionData {
  future_leak_prob?: number;
  future_leak_flag?: number;
  model_loaded: boolean;
}

export interface StreamSnapshot {
  nodes: GraphNode[];
  edges: GraphEdge[];
  metrics: StreamMetrics;
  prediction: PredictionData;
  timestamp: number;
}

export interface StreamResponse {
  data: StreamSnapshot[];
}

// Derived types for frontend usage
export interface DerivedSensor {
  id: string;
  zone_id: string;
  name: string;
  sensor_type: 'pressure' | 'flow' | 'leak_detection' | 'acoustic';
  coordinates: {
    latitude: number;
    longitude: number;
  };
  status: 'active' | 'inactive' | 'error' | 'maintenance';
  battery_level: number;
  installation_date: string;
  created_at: string;
  updated_at: string;
  last_reading: {
    id: string;
    sensor_id: string;
    value: number;
    unit: string;
    timestamp: string;
    is_anomaly: boolean;
    confidence?: number;
  };
}

export interface DerivedAlert {
  id: string;
  zone_id?: string;
  sensor_id?: string;
  node_id?: string;
  alert_type: 'leak_detected' | 'pressure_drop' | 'flow_anomaly' | 'sensor_error' | 'high_water_loss';
  severity: 'critical' | 'high' | 'medium' | 'low';
  status: 'active' | 'acknowledged' | 'resolved';
  description: string;
  location?: string;
  estimated_loss?: number;
  created_at: string;
  acknowledged_at?: string;
  resolved_at?: string;
  pressure?: number;
  leak_probability?: number;
}

export interface DerivedMetrics {
  total_water_loss: number;
  total_water_loss_percentage: number;
  critical_alerts: number;
  active_leaks: number;
  sensor_health_percentage: number;
  zones_with_issues: number;
  avg_water_loss_24h: number;
  model_accuracy?: number;
  model_precision?: number;
  model_recall?: number;
  model_f1?: number;
  prediction_prob?: number;
  prediction_flag?: number;
}

export interface NetworkData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

class BackendService {
  private lastSnapshot: StreamSnapshot | null = null;
  private alertCache: Map<string, DerivedAlert> = new Map();
  private alertIdCounter = 0;

  /**
   * Fetch the latest stream data from the backend
   */
  async fetchStream(): Promise<StreamSnapshot | null> {
    try {
      const response = await fetch(`${BACKEND_URL}/stream`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`Backend returned ${response.status}`);
      }

      const json: StreamResponse = await response.json();

      if (json.data && json.data.length > 0) {
        this.lastSnapshot = json.data[0];
        return this.lastSnapshot;
      }

      return null;
    } catch (error) {
      console.error('Failed to fetch stream from backend:', error);
      return null;
    }
  }

  /**
   * Get latest cached snapshot
   */
  getLatestSnapshot(): StreamSnapshot | null {
    return this.lastSnapshot;
  }

  /**
   * Get network data (nodes and edges) from the latest snapshot
   */
  async getNetworkData(): Promise<NetworkData | null> {
    const snapshot = await this.fetchStream();
    if (!snapshot) return null;

    return {
      nodes: snapshot.nodes,
      edges: snapshot.edges,
    };
  }

  /**
   * Transform backend nodes to sensors for the sensors page
   */
  async getSensors(): Promise<DerivedSensor[]> {
    const snapshot = await this.fetchStream();
    if (!snapshot) return [];

    // Convert backend nodes to sensor format
    // We'll treat each node as a pressure sensor and select a subset for display
    const sensors: DerivedSensor[] = snapshot.nodes.slice(0, 50).map((node, index) => {
      // Determine status based on leak status
      let status: DerivedSensor['status'] = 'active';
      if (node.is_leaking === 1) {
        status = 'error';
      } else if (node.pred_leak_prob && node.pred_leak_prob > 0.5) {
        status = 'maintenance';
      }

      // Determine sensor type based on node characteristics
      let sensorType: DerivedSensor['sensor_type'] = 'pressure';
      if (node.is_leaking === 1) {
        sensorType = 'leak_detection';
      } else if (node.pred_leak_prob && node.pred_leak_prob > 0.3) {
        sensorType = 'acoustic';
      }

      // Convert node coordinates to lat/lng (rough estimate assuming network is in a local area)
      // The backend provides x, y coordinates from WNTR which are in feet/meters
      // We'll map them to a fictional area for visualization
      const baseLat = 40.7128;
      const baseLng = -74.006;
      const scale = 0.0001; // Scale factor for coordinate conversion

      return {
        id: node.id,
        zone_id: `zone-${String.fromCharCode(65 + (index % 4))}`, // A, B, C, D
        name: `Node ${node.id}`,
        sensor_type: sensorType,
        coordinates: {
          latitude: baseLat + (node.y * scale),
          longitude: baseLng + (node.x * scale),
        },
        status,
        battery_level: Math.floor(Math.random() * 40) + 60, // 60-100%
        installation_date: '2024-01-15',
        created_at: '2024-01-15',
        updated_at: new Date().toISOString(),
        last_reading: {
          id: `reading-${node.id}`,
          sensor_id: node.id,
          value: node.pressure,
          unit: 'bar',
          timestamp: new Date().toISOString(),
          is_anomaly: node.is_leaking === 1 || (node.pred_leak_prob || 0) > 0.5,
          confidence: node.pred_leak_prob,
        },
      };
    });

    return sensors;
  }

  /**
   * Generate alerts from backend leak detection data
   */
  async getAlerts(): Promise<DerivedAlert[]> {
    const snapshot = await this.fetchStream();
    if (!snapshot) return Array.from(this.alertCache.values());

    const currentTime = new Date().toISOString();
    const newAlerts: DerivedAlert[] = [];

    // Generate alerts from nodes with leaks or high leak probability
    for (const node of snapshot.nodes) {
      const alertKey = `node-${node.id}`;

      if (node.is_leaking === 1) {
        // Actual leak detected - critical alert
        if (!this.alertCache.has(alertKey) || this.alertCache.get(alertKey)?.severity !== 'critical') {
          const alert: DerivedAlert = {
            id: `alert-${++this.alertIdCounter}`,
            node_id: node.id,
            alert_type: 'leak_detected',
            severity: 'critical',
            status: 'active',
            description: `Active leak detected at node ${node.id}`,
            location: `Node ${node.id} - Pressure: ${node.pressure.toFixed(2)} bar`,
            created_at: currentTime,
            pressure: node.pressure,
            leak_probability: node.pred_leak_prob,
          };
          this.alertCache.set(alertKey, alert);
        }
      } else if (node.pred_leak_prob && node.pred_leak_prob > 0.7) {
        // High leak probability - high severity alert
        if (!this.alertCache.has(alertKey)) {
          const alert: DerivedAlert = {
            id: `alert-${++this.alertIdCounter}`,
            node_id: node.id,
            alert_type: 'pressure_drop',
            severity: 'high',
            status: 'active',
            description: `High leak probability (${(node.pred_leak_prob * 100).toFixed(1)}%) at node ${node.id}`,
            location: `Node ${node.id}`,
            created_at: currentTime,
            pressure: node.pressure,
            leak_probability: node.pred_leak_prob,
          };
          this.alertCache.set(alertKey, alert);
        }
      } else if (node.pred_leak_prob && node.pred_leak_prob > 0.5) {
        // Medium leak probability - medium severity alert
        if (!this.alertCache.has(alertKey)) {
          const alert: DerivedAlert = {
            id: `alert-${++this.alertIdCounter}`,
            node_id: node.id,
            alert_type: 'flow_anomaly',
            severity: 'medium',
            status: 'active',
            description: `Elevated leak risk (${(node.pred_leak_prob * 100).toFixed(1)}%) at node ${node.id}`,
            location: `Node ${node.id}`,
            created_at: currentTime,
            pressure: node.pressure,
            leak_probability: node.pred_leak_prob,
          };
          this.alertCache.set(alertKey, alert);
        }
      } else {
        // No issue - remove from cache if it was there
        if (this.alertCache.has(alertKey)) {
          const existingAlert = this.alertCache.get(alertKey);
          if (existingAlert) {
            existingAlert.status = 'resolved';
            existingAlert.resolved_at = currentTime;
          }
        }
      }
    }

    // Also check edges for pipe leaks
    for (const edge of snapshot.edges) {
      const alertKey = `edge-${edge.id}`;

      if (edge.is_leaking === 1) {
        if (!this.alertCache.has(alertKey)) {
          const alert: DerivedAlert = {
            id: `alert-${++this.alertIdCounter}`,
            sensor_id: edge.id,
            alert_type: 'leak_detected',
            severity: 'critical',
            status: 'active',
            description: `Pipe leak detected on ${edge.id}`,
            location: `Pipe ${edge.id} (${edge.from} → ${edge.to})`,
            created_at: currentTime,
            leak_probability: edge.pred_leak_prob,
          };
          this.alertCache.set(alertKey, alert);
        }
      }
    }

    // Return all alerts, sorted by severity and time
    const allAlerts = Array.from(this.alertCache.values());
    const severityOrder = { critical: 0, high: 1, medium: 2, low: 3 };
    
    return allAlerts
      .filter(a => a.status === 'active')
      .sort((a, b) => {
        const severityDiff = severityOrder[a.severity] - severityOrder[b.severity];
        if (severityDiff !== 0) return severityDiff;
        return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
      })
      .slice(0, 20); // Limit to 20 most important alerts
  }

  /**
   * Get dashboard metrics from the backend data
   */
  async getMetrics(): Promise<DerivedMetrics> {
    const snapshot = await this.fetchStream();

    if (!snapshot) {
      // Return default metrics if backend is not available
      return {
        total_water_loss: 0,
        total_water_loss_percentage: 0,
        critical_alerts: 0,
        active_leaks: 0,
        sensor_health_percentage: 0,
        zones_with_issues: 0,
        avg_water_loss_24h: 0,
      };
    }

    // Calculate metrics from snapshot
    const activeLeaks = snapshot.nodes.filter(n => n.is_leaking === 1).length;
    const pipeLeaks = snapshot.edges.filter(e => e.is_leaking === 1).length;
    const totalLeaks = activeLeaks + pipeLeaks;

    const criticalAlerts = snapshot.nodes.filter(
      n => n.is_leaking === 1 || (n.pred_leak_prob || 0) > 0.7
    ).length;

    const totalNodes = snapshot.nodes.length;
    const healthyNodes = snapshot.nodes.filter(
      n => n.is_leaking === 0 && (n.pred_leak_prob || 0) < 0.3
    ).length;

    const avgPressure = snapshot.nodes.reduce((sum, n) => sum + n.pressure, 0) / totalNodes;
    const normalPressure = 30; // Expected normal pressure
    const pressureLossPercent = Math.max(0, ((normalPressure - avgPressure) / normalPressure) * 100);

    // Estimate water loss based on leak count and total flow
    const baseWaterLoss = 500; // Base loss in m³
    const leakWaterLoss = totalLeaks * 150; // Each leak adds ~150 m³
    const totalWaterLoss = baseWaterLoss + leakWaterLoss;

    // Calculate zones with issues (simplified: divide nodes into 4 zones)
    const zonesWithLeaks = new Set<number>();
    snapshot.nodes.forEach((node, i) => {
      if (node.is_leaking === 1) {
        zonesWithLeaks.add(i % 4);
      }
    });

    return {
      total_water_loss: totalWaterLoss,
      total_water_loss_percentage: Math.min(pressureLossPercent + (totalLeaks * 0.5), 100),
      critical_alerts: criticalAlerts,
      active_leaks: totalLeaks,
      sensor_health_percentage: Math.round((healthyNodes / totalNodes) * 100),
      zones_with_issues: zonesWithLeaks.size,
      avg_water_loss_24h: Math.round(totalWaterLoss * 0.9), // Slightly lower for 24h average
      model_accuracy: snapshot.metrics.nodes.accuracy,
      model_precision: snapshot.metrics.nodes.precision,
      model_recall: snapshot.metrics.nodes.recall,
      model_f1: snapshot.metrics.nodes.f1,
      prediction_prob: snapshot.prediction.future_leak_prob,
      prediction_flag: snapshot.prediction.future_leak_flag,
    };
  }

  /**
   * Get analytics data for charts
   */
  async getAnalyticsData(): Promise<{
    waterLossData: { date: string; loss: number; expected: number }[];
    zoneData: { name: string; waterLoss: number; pressure: number; flowRate: number }[];
    anomalyData: { x: number; y: number; type: string }[];
  }> {
    const snapshot = await this.fetchStream();

    if (!snapshot) {
      return {
        waterLossData: [],
        zoneData: [],
        anomalyData: [],
      };
    }

    // Generate water loss trend data (simulated based on current state)
    const now = new Date();
    const waterLossData = [];
    const baseLeaks = snapshot.nodes.filter(n => n.is_leaking === 1).length;
    
    for (let i = 12; i >= 0; i--) {
      const date = new Date(now);
      date.setDate(date.getDate() - i);
      const variation = Math.random() * 500 - 250;
      const loss = 2000 + (baseLeaks * 100) + variation;
      const expected = 2000 + Math.random() * 200;
      
      waterLossData.push({
        date: date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
        loss: Math.round(loss),
        expected: Math.round(expected),
      });
    }

    // Generate zone comparison data
    const zones = ['Zone A', 'Zone B', 'Zone C', 'Zone D'];
    const zoneData = zones.map((name, i) => {
      const zoneNodes = snapshot.nodes.filter((_, idx) => idx % 4 === i);
      const avgPressure = zoneNodes.reduce((sum, n) => sum + n.pressure, 0) / zoneNodes.length || 0;
      const leakCount = zoneNodes.filter(n => n.is_leaking === 1).length;
      
      return {
        name,
        waterLoss: 2000 + leakCount * 500 + Math.random() * 1000,
        pressure: avgPressure,
        flowRate: 100 + Math.random() * 50,
      };
    });

    // Generate anomaly detection scatter data
    const anomalyData = snapshot.nodes.slice(0, 10).map((node, i) => ({
      x: i + 1,
      y: node.pressure,
      type: node.is_leaking === 1 ? 'Anomaly' : 'Normal',
    }));

    return {
      waterLossData,
      zoneData,
      anomalyData,
    };
  }

  /**
   * Get system health status
   */
  async getSystemHealth(): Promise<{
    total_sensors: number;
    active_sensors: number;
    inactive_sensors: number;
    error_sensors: number;
    total_zones: number;
    zones_with_alerts: number;
    model_loaded: boolean;
    prediction_model_loaded: boolean;
    last_sync: string;
  }> {
    const snapshot = await this.fetchStream();

    if (!snapshot) {
      return {
        total_sensors: 0,
        active_sensors: 0,
        inactive_sensors: 0,
        error_sensors: 0,
        total_zones: 0,
        zones_with_alerts: 0,
        model_loaded: false,
        prediction_model_loaded: false,
        last_sync: new Date().toISOString(),
      };
    }

    const totalNodes = snapshot.nodes.length;
    const errorNodes = snapshot.nodes.filter(n => n.is_leaking === 1).length;
    const warningNodes = snapshot.nodes.filter(n => !n.is_leaking && (n.pred_leak_prob || 0) > 0.5).length;

    return {
      total_sensors: totalNodes,
      active_sensors: totalNodes - errorNodes - warningNodes,
      inactive_sensors: warningNodes,
      error_sensors: errorNodes,
      total_zones: 4,
      zones_with_alerts: Math.min(Math.ceil(errorNodes / 10), 4),
      model_loaded: snapshot.metrics.model_loaded,
      prediction_model_loaded: snapshot.prediction.model_loaded,
      last_sync: new Date(snapshot.timestamp * 1000).toISOString(),
    };
  }
}

// Export singleton instance
export const backendService = new BackendService();

// Export fetch function for simple use cases
export async function fetchBackendStream(): Promise<StreamSnapshot | null> {
  return backendService.fetchStream();
}
