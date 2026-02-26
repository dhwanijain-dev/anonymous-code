import { Alert } from '@/lib/types';
import { AlertCircle, CheckCircle, Clock, ChevronRight } from 'lucide-react';
import { cn } from '@/lib/utils';
import Link from 'next/link';

interface AlertFeedProps {
  alerts: Alert[];
}

export function AlertFeed({ alerts }: AlertFeedProps) {
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
    <div className="bg-white/80 backdrop-blur-sm rounded-2xl border border-slate-200 shadow-lg p-6 h-full">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-slate-900">Recent Alerts</h3>
        <Link href="/alerts" className="text-sm text-slate-600 hover:text-slate-900 flex items-center gap-1 transition-colors">
          View all <ChevronRight className="w-4 h-4" />
        </Link>
      </div>
      <div className="space-y-3">
        {alerts.map((alert) => {
          const styles = getSeverityStyles(alert.severity);
          return (
            <div
              key={alert.id}
              className={cn(
                'p-4 rounded-xl bg-gradient-to-r border transition-all duration-300 hover:scale-[1.01] cursor-pointer group shadow-sm',
                styles.bg,
                styles.border
              )}
            >
              <div className="flex items-start gap-3">
                <div className={cn('mt-0.5', styles.text)}>{getStatusIcon(alert.status)}</div>
                <div className="flex-1 min-w-0">
                  <p className="font-medium text-sm text-slate-900 truncate group-hover:text-slate-700">{alert.description}</p>
                  {alert.location && (
                    <p className="text-xs text-slate-500 mt-1">{alert.location}</p>
                  )}
                  <div className="flex items-center gap-2 mt-3">
                    <span className={cn('text-xs px-2 py-1 rounded-lg font-medium capitalize', styles.badge)}>
                      {alert.severity}
                    </span>
                    <span className="text-xs text-slate-400">
                      {formatTime(alert.created_at)}
                    </span>
                  </div>
                </div>
                <ChevronRight className="w-4 h-4 text-slate-300 group-hover:text-slate-500 group-hover:translate-x-1 transition-all" />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
