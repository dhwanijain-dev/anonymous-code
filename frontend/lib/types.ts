// Water Leakage Detection System Types

export interface User {
  id: string;
  email: string;
  name: string;
  role: 'admin' | 'technician' | 'supervisor' | 'viewer';
  phone?: string;
  zone_id?: string;
  avatar_url?: string;
  created_at: string;
  updated_at: string;
}

export interface Zone {
  id: string;
  name: string;
  location: string;
  coordinates: {
    latitude: number;
    longitude: number;
  };
  boundary?: {
    type: 'Feature';
    geometry: {
      type: 'Polygon';
      coordinates: number[][][];
    };
  };
  status: 'active' | 'inactive' | 'maintenance';
  total_sensors: number;
  active_sensors: number;
  water_loss_percentage: number;
  created_at: string;
  updated_at: string;
}

export interface Sensor {
  id: string;
  zone_id: string;
  name: string;
  sensor_type: 'pressure' | 'flow' | 'leak_detection' | 'acoustic';
  coordinates: {
    latitude: number;
    longitude: number;
  };
  status: 'active' | 'inactive' | 'error' | 'maintenance';
  last_reading?: SensorReading;
  battery_level?: number;
  installation_date: string;
  created_at: string;
  updated_at: string;
}

export interface SensorReading {
  id: string;
  sensor_id: string;
  value: number;
  unit: string;
  timestamp: string;
  is_anomaly: boolean;
  confidence?: number;
}

export interface Alert {
  id: string;
  zone_id?: string;
  sensor_id?: string;
  alert_type: 'leak_detected' | 'pressure_drop' | 'flow_anomaly' | 'sensor_error' | 'high_water_loss';
  severity: 'critical' | 'high' | 'medium' | 'low';
  status: 'active' | 'acknowledged' | 'resolved';
  description: string;
  location?: string;
  estimated_loss?: number;
  created_at: string;
  acknowledged_at?: string;
  resolved_at?: string;
  assigned_to?: string;
  notes?: string;
}

export interface AlertThreshold {
  id: string;
  zone_id: string;
  sensor_type: string;
  alert_type: string;
  threshold_value: number;
  comparison: 'greater_than' | 'less_than' | 'equal_to' | 'between';
  upper_value?: number;
  enabled: boolean;
  created_at: string;
  updated_at: string;
}

export interface AnalyticsData {
  timestamp: string;
  zone_id: string;
  total_flow: number;
  expected_flow: number;
  water_loss: number;
  water_loss_percentage: number;
  pressure_avg: number;
  pressure_min: number;
  pressure_max: number;
  leak_count: number;
}

export interface DashboardMetrics {
  total_water_loss: number;
  total_water_loss_percentage: number;
  critical_alerts: number;
  active_leaks: number;
  sensor_health_percentage: number;
  zones_with_issues: number;
  avg_water_loss_24h: number;
}

export interface NotificationSetting {
  id: string;
  user_id: string;
  alert_type: string;
  severity: string;
  method: 'email' | 'sms' | 'push' | 'in_app';
  enabled: boolean;
  created_at: string;
  updated_at: string;
}

export interface SystemHealth {
  total_sensors: number;
  active_sensors: number;
  inactive_sensors: number;
  error_sensors: number;
  total_zones: number;
  zones_with_alerts: number;
  api_uptime_percentage: number;
  last_sync: string;
}
