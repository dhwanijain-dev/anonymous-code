import { TrendingUp, TrendingDown } from 'lucide-react';
import { cn } from '@/lib/utils';
import { LucideIcon } from 'lucide-react';

interface KPICardProps {
  title: string;
  value: string | number;
  change: number;
  unit?: string;
  icon: LucideIcon;
  trend?: 'up' | 'down';
  status?: 'success' | 'warning' | 'critical' | 'info';
}

export function KPICard({
  title,
  value,
  change,
  unit,
  icon: Icon,
  trend = 'up',
  status = 'info',
}: KPICardProps) {
  const statusStyles = {
    success: {
      bg: 'from-emerald-50 to-green-50 border-emerald-200',
      icon: 'from-emerald-500 to-green-600',
      glow: 'shadow-emerald-200',
      text: 'text-emerald-600',
    },
    warning: {
      bg: 'from-amber-50 to-orange-50 border-amber-200',
      icon: 'from-amber-500 to-orange-600',
      glow: 'shadow-amber-200',
      text: 'text-amber-600',
    },
    critical: {
      bg: 'from-red-50 to-rose-50 border-red-200',
      icon: 'from-red-500 to-rose-600',
      glow: 'shadow-red-200',
      text: 'text-red-600',
    },
    info: {
      bg: 'from-blue-50 to-indigo-50 border-blue-200',
      icon: 'from-slate-700 to-slate-900',
      glow: 'shadow-slate-300',
      text: 'text-blue-600',
    },
  };

  const currentStatus = statusStyles[status];
  const trendColor = trend === 'up' ? 'text-red-600' : 'text-emerald-600';

  return (
    <div className={cn(
      'relative overflow-hidden bg-gradient-to-br rounded-2xl border p-6 transition-all duration-300 hover:scale-[1.02] hover:shadow-xl group bg-white/80 backdrop-blur-sm',
      currentStatus.bg
    )}>
      {/* Background glow effect */}
      <div className={cn(
        'absolute -top-12 -right-12 w-32 h-32 rounded-full blur-3xl opacity-20 group-hover:opacity-30 transition-opacity',
        `bg-gradient-to-br ${currentStatus.icon}`
      )} />
      
      <div className="relative flex items-start justify-between">
        <div className="space-y-3">
          <p className="text-sm text-slate-500 font-medium">{title}</p>
          <p className="text-3xl font-bold text-slate-900">{value}</p>
          {unit && (
            <div className="flex items-center gap-1.5 text-sm">
              {trend === 'up' ? (
                <TrendingUp className={cn('w-4 h-4', trendColor)} />
              ) : (
                <TrendingDown className={cn('w-4 h-4', trendColor)} />
              )}
              <span className={trendColor}>
                {trend === 'up' ? '+' : ''}{change} {unit}
              </span>
            </div>
          )}
        </div>
        <div className={cn(
          'p-3 rounded-xl bg-gradient-to-br shadow-lg',
          currentStatus.icon,
          currentStatus.glow
        )}>
          <Icon className="w-6 h-6 text-white" />
        </div>
      </div>
    </div>
  );
}
