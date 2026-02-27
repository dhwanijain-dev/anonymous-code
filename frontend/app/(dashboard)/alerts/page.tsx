'use client';

import { useState, useEffect, useCallback } from 'react';
import Link from 'next/link';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { AlertCircle, CheckCircle, Clock, AlertTriangle, ChevronRight, Search, Filter, RefreshCw } from 'lucide-react';
import { Alert } from '@/lib/types';
import { cn } from '@/lib/utils';
import { backendService, DerivedAlert } from '@/lib/backend-service';

export default function AlertsPage() {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterSeverity, setFilterSeverity] = useState<string>('all');
  const [filterStatus, setFilterStatus] = useState<string>('all');

  // Fetch alerts from backend
  const fetchAlerts = useCallback(async () => {
    try {
      const backendAlerts = await backendService.getAlerts();
      const mappedAlerts: Alert[] = backendAlerts.map((a: DerivedAlert) => ({
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
      }));
      setAlerts(mappedAlerts);
      setIsLoading(false);
    } catch (error) {
      console.error('Failed to fetch alerts:', error);
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchAlerts();
    const interval = setInterval(fetchAlerts, 5000);
    return () => clearInterval(interval);
  }, [fetchAlerts]);

  const filteredAlerts = alerts.filter((alert) => {
    const matchSearch = alert.description.toLowerCase().includes(searchTerm.toLowerCase());
    const matchSeverity = filterSeverity === 'all' || alert.severity === filterSeverity;
    const matchStatus = filterStatus === 'all' || alert.status === filterStatus;
    return matchSearch && matchSeverity && matchStatus;
  });

  const getSeverityStyles = (severity: Alert['severity']) => {
    switch (severity) {
      case 'critical':
        return { bg: 'from-red-50 to-red-100/50', border: 'border-red-200', text: 'text-red-600', badge: 'bg-red-100 text-red-700' };
      case 'high':
        return { bg: 'from-amber-50 to-amber-100/50', border: 'border-amber-200', text: 'text-amber-600', badge: 'bg-amber-100 text-amber-700' };
      case 'medium':
        return { bg: 'from-blue-50 to-blue-100/50', border: 'border-blue-200', text: 'text-blue-600', badge: 'bg-blue-100 text-blue-700' };
      case 'low':
        return { bg: 'from-emerald-50 to-emerald-100/50', border: 'border-emerald-200', text: 'text-emerald-600', badge: 'bg-emerald-100 text-emerald-700' };
    }
  };

  const getStatusIcon = (status: Alert['status']) => {
    switch (status) {
      case 'active':
        return <AlertCircle className="w-5 h-5" />;
      case 'acknowledged':
        return <Clock className="w-5 h-5" />;
      case 'resolved':
        return <CheckCircle className="w-5 h-5" />;
    }
  };

  const formatTime = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);

    if (minutes < 1) return 'now';
    if (minutes < 60) return `${minutes}m ago`;
    if (hours < 24) return `${hours}h ago`;
    return date.toLocaleDateString();
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-slate-900">Alerts</h1>
          <p className="text-slate-500 mt-1">Monitor and manage system alerts (Live from simulation)</p>
        </div>
        <div className="flex items-center gap-3">
          <Button
            onClick={fetchAlerts}
            variant="outline"
            size="sm"
            className="flex items-center gap-2"
          >
            <RefreshCw className={cn("w-4 h-4", isLoading && "animate-spin")} />
            Refresh
          </Button>
          <div className="flex items-center gap-2 px-4 py-2 bg-red-50 text-red-700 rounded-xl text-sm font-medium border border-red-200">
            <AlertCircle className="w-4 h-4" />
            {alerts.filter((a) => a.status === 'active').length} Active
          </div>
        </div>
      </div>

      {/* Filters */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="relative">
          <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
          <Input
            placeholder="Search alerts..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-11 h-11 bg-slate-50 border-slate-200 text-slate-900 placeholder:text-slate-400 rounded-xl focus:bg-white focus:border-slate-400"
          />
        </div>
        <Select value={filterSeverity} onValueChange={setFilterSeverity}>
          <SelectTrigger className="h-11 bg-slate-50 border-slate-200 text-slate-900 rounded-xl">
            <SelectValue placeholder="Filter by severity" />
          </SelectTrigger>
          <SelectContent className="bg-white border-slate-200 shadow-lg">
            <SelectItem value="all">All Severity</SelectItem>
            <SelectItem value="critical">Critical</SelectItem>
            <SelectItem value="high">High</SelectItem>
            <SelectItem value="medium">Medium</SelectItem>
            <SelectItem value="low">Low</SelectItem>
          </SelectContent>
        </Select>
        <Select value={filterStatus} onValueChange={setFilterStatus}>
          <SelectTrigger className="h-11 bg-slate-50 border-slate-200 text-slate-900 rounded-xl">
            <SelectValue placeholder="Filter by status" />
          </SelectTrigger>
          <SelectContent className="bg-white border-slate-200 shadow-lg">
            <SelectItem value="all">All Status</SelectItem>
            <SelectItem value="active">Active</SelectItem>
            <SelectItem value="acknowledged">Acknowledged</SelectItem>
            <SelectItem value="resolved">Resolved</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* Alerts List */}
      <div className="space-y-3">
        {filteredAlerts.map((alert) => {
          const styles = getSeverityStyles(alert.severity);
          return (
            <Link key={alert.id} href={`/alerts/${alert.id}`}>
              <div className={cn(
                'p-5 rounded-2xl bg-gradient-to-r border shadow-sm transition-all duration-300 hover:scale-[1.01] cursor-pointer group',
                styles.bg,
                styles.border
              )}>
                <div className="flex items-start justify-between gap-4">
                  <div className="flex items-start gap-4 flex-1">
                    <div className={cn('mt-0.5', styles.text)}>
                      {getStatusIcon(alert.status)}
                    </div>
                    <div className="flex-1 min-w-0">
                      <h3 className="font-semibold text-slate-900 group-hover:text-slate-700">{alert.description}</h3>
                      {alert.location && (
                        <p className="text-sm text-slate-500 mt-1">{alert.location}</p>
                      )}
                      <div className="flex items-center gap-2 mt-3">
                        <span className="text-xs px-2.5 py-1 bg-slate-100 rounded-lg text-slate-600">
                          {alert.alert_type.replace('_', ' ')}
                        </span>
                        <span className="text-xs text-slate-400">
                          {formatTime(alert.created_at)}
                        </span>
                      </div>
                    </div>
                  </div>
                  <div className="flex flex-col items-end gap-2">
                    <span className={cn('px-3 py-1.5 rounded-lg text-xs font-bold capitalize', styles.badge)}>
                      {alert.severity}
                    </span>
                    <span className="text-xs text-slate-500 capitalize">
                      {alert.status}
                    </span>
                    {alert.estimated_loss && (
                      <span className="text-xs font-medium text-amber-600">
                        {alert.estimated_loss} m³/h
                      </span>
                    )}
                    <ChevronRight className="w-4 h-4 text-slate-300 group-hover:text-slate-500 group-hover:translate-x-1 transition-all mt-1" />
                  </div>
                </div>
              </div>
            </Link>
          );
        })}
      </div>

      {filteredAlerts.length === 0 && (
        <div className="bg-white/80 backdrop-blur-sm rounded-2xl border border-slate-200 shadow-lg p-12 text-center">
          <AlertTriangle className="w-12 h-12 text-slate-300 mx-auto mb-3" />
          <p className="text-slate-500">No alerts found matching your filters</p>
        </div>
      )}
    </div>
  );
}
