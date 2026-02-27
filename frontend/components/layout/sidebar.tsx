'use client';

import { useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { User } from '@/lib/types';
import {
  LayoutDashboard,
  Map,
  Droplet,
  AlertCircle,
  BarChart3,
  Settings,
  Menu,
  X,
  Users,
  Zap,
  Waves,
} from 'lucide-react';
import { cn } from '@/lib/utils';

const navigationItems = [
  {
    label: 'Dashboard',
    href: '/dashboard',
    icon: LayoutDashboard,
    roles: ['admin', 'technician', 'supervisor', 'viewer'],
  },
  {
    label: 'Live Map',
    href: '/live-map',
    icon: Map,
    roles: ['admin', 'technician', 'supervisor', 'viewer'],
  },
  {
    label: 'Sensors',
    href: '/sensors',
    icon: Droplet,
    roles: ['admin', 'technician', 'supervisor', 'viewer'],
  },
  {
    label: 'Alerts',
    href: '/alerts',
    icon: AlertCircle,
    roles: ['admin', 'technician', 'supervisor', 'viewer'],
  },
  {
    label: 'Analytics',
    href: '/analytics',
    icon: BarChart3,
    roles: ['admin', 'technician', 'supervisor'],
  },
  {
    label: 'Admin',
    href: '/admin',
    icon: Settings,
    roles: ['admin'],
  },
];

interface SidebarProps {
  user: User | null;
}

export function Sidebar({ user }: SidebarProps) {
  const pathname = usePathname();
  const [isOpen, setIsOpen] = useState(false);

  const filteredItems = navigationItems.filter(
    (item) => !user || item.roles.includes(user.role)
  );

  const handleNavigation = () => {
    if (window.innerWidth < 768) {
      setIsOpen(false);
    }
  };

  return (
    <>
      {/* Mobile menu button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="md:hidden fixed top-4 left-4 z-40 p-2.5 bg-white border border-slate-200 rounded-xl hover:bg-slate-50 text-slate-700 transition-all duration-300 shadow-lg"
      >
        {isOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
      </button>

      {/* Sidebar */}
      <div
        className={cn(
          'fixed md:relative w-72 h-screen bg-white/80 backdrop-blur-xl border-r border-slate-200 flex flex-col transition-all duration-300 z-30 shadow-xl md:shadow-none',
          isOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'
        )}
      >
        {/* Logo */}
        <div className="p-6 border-b border-slate-200">
          <Link href="/dashboard" className="flex items-center gap-3 group">
            <div className="relative">
              <div className="absolute inset-0 bg-slate-800/20 blur-xl rounded-full group-hover:bg-slate-800/30 transition-all" />
              <div className="relative bg-linear-to-br from-slate-800 to-slate-900 p-2.5 rounded-xl shadow-lg">
                 <img src="/logo.svg" alt="LeakNet Logo" height={ 10} width={10} />
              </div>
            </div>
            <div>
              <span className="text-xl font-bold text-slate-900">LeakNet</span>
              <div className="flex items-center gap-1 text-slate-500 text-xs">
                <Waves className="w-3 h-3" />
                <span>Water Monitoring</span>
              </div>
            </div>
          </Link>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-4 space-y-1.5 overflow-y-auto">
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider px-4 mb-3">Navigation</p>
          {filteredItems.map((item) => {
            const Icon = item.icon;
            const isActive = pathname === item.href || pathname.startsWith(item.href + '/');

            return (
              <Link
                key={item.href}
                href={item.href}
                onClick={handleNavigation}
                className={cn(
                  'flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-300 font-medium text-sm group',
                  isActive
                    ? 'bg-gradient-to-r from-slate-800 to-slate-900 text-white shadow-lg'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
                )}
              >
                <div className={cn(
                  'p-2 rounded-lg transition-all duration-300',
                  isActive
                    ? 'bg-white/20'
                    : 'bg-slate-100 group-hover:bg-slate-200'
                )}>
                  <Icon className="w-4 h-4" />
                </div>
                <span>{item.label}</span>
                {isActive && (
                  <div className="ml-auto w-1.5 h-1.5 bg-white rounded-full" />
                )}
              </Link>
            );
          })}
        </nav>

        {/* User profile */}
        {user && (
          <div className="p-4 border-t border-slate-200">
            <div className="flex items-center gap-3 p-3 rounded-xl bg-slate-50 border border-slate-200">
              <div className="relative">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-slate-700 to-slate-900 flex items-center justify-center shadow-lg">
                  <span className="text-sm font-bold text-white">
                    {user.name.charAt(0).toUpperCase()}
                  </span>
                </div>
                <div className="absolute -bottom-0.5 -right-0.5 w-3 h-3 bg-emerald-500 rounded-full border-2 border-white" />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-slate-900 truncate">{user.name}</p>
                <p className="text-xs text-slate-500 truncate capitalize">{user.role}</p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Mobile overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/20 backdrop-blur-sm z-20 md:hidden"
          onClick={() => setIsOpen(false)}
        />
      )}
    </>
  );
}
