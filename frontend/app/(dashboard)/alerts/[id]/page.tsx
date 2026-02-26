'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { ArrowLeft, AlertCircle, CheckCircle, Clock, MapPin, Zap, Calendar } from 'lucide-react';
import { Alert } from '@/lib/types';
import { cn } from '@/lib/utils';

// Mock data
const mockAlert: Alert = {
  id: '1',
  zone_id: 'zone-a',
  alert_type: 'leak_detected',
  severity: 'critical',
  status: 'active',
  description: 'High-pressure leak detected in Zone A-2',
  location: 'Main Pipeline - Sector 2',
  estimated_loss: 850,
  created_at: new Date(Date.now() - 5 * 60000).toISOString(),
  notes: 'Requires immediate inspection. High water pressure detected at junction point.',
};

interface AlertDetailPageProps {
  params: Promise<{ id: string }>;
}

export default function AlertDetailPage({ params }: AlertDetailPageProps) {
  const router = useRouter();
  const [status, setStatus] = useState<Alert['status']>(mockAlert.status);
  const [notes, setNotes] = useState(mockAlert.notes || '');

  const getSeverityColor = (severity: Alert['severity']) => {
    switch (severity) {
      case 'critical':
        return 'bg-destructive/20 text-destructive border-destructive/30';
      case 'high':
        return 'bg-warning/20 text-warning border-warning/30';
      case 'medium':
        return 'bg-info/20 text-info border-info/30';
      case 'low':
        return 'bg-success/20 text-success border-success/30';
    }
  };

  const getStatusIcon = (st: Alert['status']) => {
    switch (st) {
      case 'active':
        return <AlertCircle className="w-6 h-6" />;
      case 'acknowledged':
        return <Clock className="w-6 h-6" />;
      case 'resolved':
        return <CheckCircle className="w-6 h-6" />;
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
        <div className="flex items-center gap-4">
          <Button
            variant="ghost"
            size="icon"
            onClick={() => router.back()}
          >
            <ArrowLeft className="w-5 h-5" />
          </Button>
          <div>
            <h1 className="text-3xl font-bold">{mockAlert.description}</h1>
            <p className="text-muted-foreground mt-1">Alert ID: {mockAlert.id}</p>
          </div>
        </div>
        <div className={cn('p-4 rounded-lg', getSeverityColor(mockAlert.severity))}>
          {getStatusIcon(status)}
        </div>
      </div>

      {/* Status and Actions */}
      <div className="flex gap-2 flex-wrap">
        <Button variant={status === 'active' ? 'default' : 'outline'} onClick={() => setStatus('active')}>
          Mark Active
        </Button>
        <Button variant={status === 'acknowledged' ? 'default' : 'outline'} onClick={() => setStatus('acknowledged')}>
          Acknowledge
        </Button>
        <Button variant={status === 'resolved' ? 'default' : 'outline'} onClick={() => setStatus('resolved')}>
          Resolve
        </Button>
      </div>

      {/* Alert Details */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main details */}
        <div className="lg:col-span-2 space-y-6">
          {/* Information card */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Alert Information</CardTitle>
            </CardHeader>
            <CardContent className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <p className="text-sm text-muted-foreground flex items-center gap-2">
                  <AlertCircle className="w-4 h-4" />
                  Alert Type
                </p>
                <p className="font-medium mt-2 text-sm capitalize">{mockAlert.alert_type.replace('_', ' ')}</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground flex items-center gap-2">
                  <Zap className="w-4 h-4" />
                  Severity
                </p>
                <div className={cn('inline-block px-2 py-1 rounded-lg text-sm font-bold capitalize mt-2', getSeverityColor(mockAlert.severity))}>
                  {mockAlert.severity}
                </div>
              </div>
              <div>
                <p className="text-sm text-muted-foreground flex items-center gap-2">
                  <MapPin className="w-4 h-4" />
                  Location
                </p>
                <p className="font-medium mt-2 text-sm">{mockAlert.location}</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground flex items-center gap-2">
                  <Calendar className="w-4 h-4" />
                  Time Since Alert
                </p>
                <p className="font-medium mt-2 text-sm">{formatTime(mockAlert.created_at)}</p>
              </div>
              {mockAlert.zone_id && (
                <div>
                  <p className="text-sm text-muted-foreground">Zone</p>
                  <p className="font-medium mt-2 text-sm uppercase">{mockAlert.zone_id}</p>
                </div>
              )}
              {mockAlert.estimated_loss && (
                <div>
                  <p className="text-sm text-muted-foreground">Estimated Water Loss</p>
                  <p className="font-medium mt-2 text-sm">{mockAlert.estimated_loss} m³/h</p>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Timeline */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Timeline</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex gap-4">
                  <div className="flex flex-col items-center">
                    <AlertCircle className="w-6 h-6 text-destructive" />
                    <div className="w-0.5 h-12 bg-border mt-2"></div>
                  </div>
                  <div>
                    <p className="font-medium">Alert Created</p>
                    <p className="text-sm text-muted-foreground">
                      {new Date(mockAlert.created_at).toLocaleString()}
                    </p>
                  </div>
                </div>
                {status === 'acknowledged' && (
                  <div className="flex gap-4">
                    <div className="flex flex-col items-center">
                      <Clock className="w-6 h-6 text-warning" />
                      <div className="w-0.5 h-12 bg-border mt-2"></div>
                    </div>
                    <div>
                      <p className="font-medium">Alert Acknowledged</p>
                      <p className="text-sm text-muted-foreground">
                        {new Date().toLocaleString()}
                      </p>
                    </div>
                  </div>
                )}
                {status === 'resolved' && (
                  <div className="flex gap-4">
                    <div className="flex flex-col items-center">
                      <CheckCircle className="w-6 h-6 text-success" />
                    </div>
                    <div>
                      <p className="font-medium">Alert Resolved</p>
                      <p className="text-sm text-muted-foreground">
                        {new Date().toLocaleString()}
                      </p>
                    </div>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Notes */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Notes</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <Textarea
                placeholder="Add notes about this alert..."
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                className="min-h-32"
              />
              <Button>Save Notes</Button>
            </CardContent>
          </Card>
        </div>

        {/* Sidebar */}
        <div className="space-y-4">
          {/* Status card */}
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Current Status</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center gap-2">
                {status === 'active' && <AlertCircle className="w-5 h-5 text-destructive" />}
                {status === 'acknowledged' && <Clock className="w-5 h-5 text-warning" />}
                {status === 'resolved' && <CheckCircle className="w-5 h-5 text-success" />}
                <span className="font-medium capitalize">{status}</span>
              </div>
            </CardContent>
          </Card>

          {/* Risk Assessment */}
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Risk Assessment</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div>
                <p className="text-xs text-muted-foreground">Severity Level</p>
                <div className="w-full bg-border rounded-full h-2 mt-1 overflow-hidden">
                  <div className="bg-destructive h-full" style={{ width: '95%' }}></div>
                </div>
                <p className="text-xs font-medium mt-1">Critical</p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Impact Score</p>
                <p className="text-lg font-bold text-destructive">8.5/10</p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Response Time</p>
                <p className="text-sm font-medium">Immediate action required</p>
              </div>
            </CardContent>
          </Card>

          {/* Related Resources */}
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Related Resources</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              <Button variant="outline" className="w-full justify-start text-sm">
                View Zone Details
              </Button>
              <Button variant="outline" className="w-full justify-start text-sm">
                View Sensor Data
              </Button>
              <Button variant="outline" className="w-full justify-start text-sm">
                Historical Alerts
              </Button>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
