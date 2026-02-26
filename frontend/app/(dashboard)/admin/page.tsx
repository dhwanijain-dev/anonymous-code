'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Users, MapPin, Settings as SettingsIcon, Activity, Shield, Database, Server, ChevronRight } from 'lucide-react';
import { User, Zone } from '@/lib/types';
import { cn } from '@/lib/utils';

// Mock data
const mockUsers: User[] = [
  {
    id: '1',
    email: 'admin@aquaguard.com',
    name: 'Admin User',
    role: 'admin',
    phone: '+1 (555) 123-4567',
    created_at: '2024-01-01',
    updated_at: '2024-02-26',
  },
  {
    id: '2',
    email: 'tech@aquaguard.com',
    name: 'Tech Support',
    role: 'technician',
    phone: '+1 (555) 234-5678',
    created_at: '2024-01-10',
    updated_at: '2024-02-26',
  },
  {
    id: '3',
    email: 'supervisor@aquaguard.com',
    name: 'Operations Supervisor',
    role: 'supervisor',
    phone: '+1 (555) 345-6789',
    created_at: '2024-01-15',
    updated_at: '2024-02-26',
  },
];

const mockZones: Zone[] = [
  {
    id: 'zone-a',
    name: 'Downtown District',
    location: 'Central City',
    coordinates: { latitude: 40.7128, longitude: -74.006 },
    status: 'active',
    total_sensors: 12,
    active_sensors: 11,
    water_loss_percentage: 12.5,
    created_at: '2024-01-01',
    updated_at: '2024-02-26',
  },
  {
    id: 'zone-b',
    name: 'Suburban Area',
    location: 'North Side',
    coordinates: { latitude: 40.7138, longitude: -74.016 },
    status: 'active',
    total_sensors: 8,
    active_sensors: 8,
    water_loss_percentage: 8.2,
    created_at: '2024-01-05',
    updated_at: '2024-02-26',
  },
  {
    id: 'zone-c',
    name: 'Industrial Complex',
    location: 'East Side',
    coordinates: { latitude: 40.7118, longitude: -73.996 },
    status: 'active',
    total_sensors: 15,
    active_sensors: 14,
    water_loss_percentage: 6.8,
    created_at: '2024-01-10',
    updated_at: '2024-02-26',
  },
];

const tabs = [
  { id: 'overview', label: 'Overview', icon: Activity },
  { id: 'users', label: 'Users', icon: Users },
  { id: 'zones', label: 'Zones', icon: MapPin },
  { id: 'settings', label: 'Settings', icon: SettingsIcon },
] as const;

export default function AdminPage() {
  const [selectedTab, setSelectedTab] = useState<'overview' | 'users' | 'zones' | 'settings'>('overview');

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-slate-900">Administration</h1>
        <p className="text-slate-500 mt-1">System configuration and management</p>
      </div>

      {/* Custom Tab Navigation */}
      <div className="bg-white/80 backdrop-blur-sm rounded-2xl border border-slate-200 shadow-sm p-1.5 flex gap-1">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          return (
            <button
              key={tab.id}
              onClick={() => setSelectedTab(tab.id)}
              className={cn(
                'flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl text-sm font-medium transition-all duration-300',
                selectedTab === tab.id
                  ? 'bg-linear-to-r from-slate-800 to-slate-900 text-white shadow-lg'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
              )}
            >
              <Icon className="w-4 h-4" />
              {tab.label}
            </button>
          );
        })}
      </div>

      {/* Overview Tab */}
      {selectedTab === 'overview' && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-linear-to-br from-blue-50 to-indigo-50 rounded-2xl border border-blue-200 shadow-sm p-6">
              <div className="flex items-start justify-between">
                <div>
                  <p className="text-sm text-slate-600">Total Users</p>
                  <p className="text-3xl font-bold text-slate-900 mt-2">{mockUsers.length}</p>
                  <p className="text-xs text-slate-500 mt-2">Active accounts</p>
                </div>
                <div className="p-3 rounded-xl bg-linear-to-br from-blue-400 to-indigo-500 shadow-lg">
                  <Users className="w-6 h-6 text-white" />
                </div>
              </div>
            </div>
            <div className="bg-linear-to-br from-emerald-50 to-teal-50 rounded-2xl border border-emerald-200 shadow-sm p-6">
              <div className="flex items-start justify-between">
                <div>
                  <p className="text-sm text-slate-600">Monitored Zones</p>
                  <p className="text-3xl font-bold text-slate-900 mt-2">{mockZones.length}</p>
                  <p className="text-xs text-slate-500 mt-2">All active</p>
                </div>
                <div className="p-3 rounded-xl bg-linear-to-br from-emerald-400 to-teal-500 shadow-lg">
                  <MapPin className="w-6 h-6 text-white" />
                </div>
              </div>
            </div>
            <div className="bg-linear-to-br from-violet-50 to-purple-50 rounded-2xl border border-violet-200 shadow-sm p-6">
              <div className="flex items-start justify-between">
                <div>
                  <p className="text-sm text-slate-600">System Uptime</p>
                  <p className="text-3xl font-bold text-slate-900 mt-2">99.8%</p>
                  <p className="text-xs text-emerald-600 mt-2">Last 30 days</p>
                </div>
                <div className="p-3 rounded-xl bg-linear-to-br from-violet-400 to-purple-500 shadow-lg">
                  <Server className="w-6 h-6 text-white" />
                </div>
              </div>
            </div>
          </div>

          {/* System Health */}
          <div className="bg-white/80 backdrop-blur-sm rounded-2xl border border-slate-200 shadow-lg p-6">
            <h3 className="text-lg font-semibold text-slate-900 flex items-center gap-2 mb-6">
              <Activity className="w-5 h-5 text-blue-600" />
              System Health
            </h3>
            <div className="space-y-4">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-slate-900">API Server</span>
                  <span className="text-sm text-emerald-600">Healthy</span>
                </div>
                <div className="w-full bg-slate-100 rounded-full h-2 overflow-hidden">
                  <div className="bg-linear-to-r from-emerald-400 to-teal-400 h-full rounded-full" style={{ width: '100%' }}></div>
                </div>
              </div>
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-slate-900">Database</span>
                  <span className="text-sm text-emerald-600">Healthy</span>
                </div>
                <div className="w-full bg-slate-100 rounded-full h-2 overflow-hidden">
                  <div className="bg-linear-to-r from-emerald-400 to-teal-400 h-full rounded-full" style={{ width: '100%' }}></div>
                </div>
              </div>
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-slate-900">Sensor Network</span>
                  <span className="text-sm text-emerald-600">Healthy</span>
                </div>
                <div className="w-full bg-slate-100 rounded-full h-2 overflow-hidden">
                  <div className="bg-linear-to-r from-emerald-400 to-teal-400 h-full rounded-full" style={{ width: '97%' }}></div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Users Tab */}
      {selectedTab === 'users' && (
        <div className="space-y-6">
          <div className="flex justify-between items-center">
            <h2 className="text-xl font-semibold text-slate-900 flex items-center gap-2">
              <Users className="w-5 h-5 text-blue-600" />
              User Management
            </h2>
            <Button className="bg-linear-to-r from-slate-800 to-slate-900 text-white hover:opacity-90 rounded-xl">
              Add User
            </Button>
          </div>

          <div className="bg-white/80 backdrop-blur-sm rounded-2xl border border-slate-200 shadow-lg overflow-hidden">
            <table className="w-full text-sm">
              <thead className="border-b border-slate-200 bg-slate-50">
                <tr>
                  <th className="px-4 py-4 text-left font-medium text-slate-600">Name</th>
                  <th className="px-4 py-4 text-left font-medium text-slate-600">Email</th>
                  <th className="px-4 py-4 text-left font-medium text-slate-600">Role</th>
                  <th className="px-4 py-4 text-left font-medium text-slate-600">Phone</th>
                  <th className="px-4 py-4 text-left font-medium text-slate-600">Joined</th>
                  <th className="px-4 py-4 text-left font-medium text-slate-600">Actions</th>
                </tr>
              </thead>
              <tbody>
                {mockUsers.map((user, idx) => (
                  <tr key={user.id} className={cn(
                    "border-b border-slate-100 hover:bg-slate-50 transition-colors",
                    idx === mockUsers.length - 1 && "border-b-0"
                  )}>
                    <td className="px-4 py-4 font-medium text-slate-900">{user.name}</td>
                    <td className="px-4 py-4 text-slate-600">{user.email}</td>
                    <td className="px-4 py-4">
                      <span className={cn(
                        "px-2.5 py-1 rounded-lg text-xs font-medium",
                        user.role === 'admin' && "bg-violet-100 text-violet-700 border border-violet-200",
                        user.role === 'technician' && "bg-blue-100 text-blue-700 border border-blue-200",
                        user.role === 'supervisor' && "bg-amber-100 text-amber-700 border border-amber-200"
                      )}>
                        {user.role}
                      </span>
                    </td>
                    <td className="px-4 py-4 text-slate-600">{user.phone}</td>
                    <td className="px-4 py-4 text-slate-500 text-xs">
                      {new Date(user.created_at).toLocaleDateString()}
                    </td>
                    <td className="px-4 py-4">
                      <Button size="sm" className="bg-slate-100 border border-slate-200 text-slate-700 hover:bg-slate-200 rounded-lg">
                        Edit
                      </Button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Zones Tab */}
      {selectedTab === 'zones' && (
        <div className="space-y-6">
          <div className="flex justify-between items-center">
            <h2 className="text-xl font-semibold text-slate-900 flex items-center gap-2">
              <MapPin className="w-5 h-5 text-blue-600" />
              Zone Management
            </h2>
            <Button className="bg-linear-to-r from-slate-800 to-slate-900 text-white hover:opacity-90 rounded-xl">
              Add Zone
            </Button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {mockZones.map((zone) => (
              <div key={zone.id} className="bg-white/80 backdrop-blur-sm rounded-2xl border border-slate-200 shadow-sm p-5 hover:bg-white hover:shadow-md transition-all duration-300 group">
                <div className="flex items-start justify-between mb-4">
                  <h3 className="font-semibold text-slate-900">{zone.name}</h3>
                  <span className={cn(
                    "px-2 py-1 rounded-lg text-xs font-medium",
                    zone.status === 'active'
                      ? 'bg-emerald-100 text-emerald-700 border border-emerald-200'
                      : 'bg-slate-100 text-slate-500 border border-slate-200'
                  )}>
                    {zone.status}
                  </span>
                </div>
                <div className="space-y-3 mb-4">
                  <div className="flex justify-between">
                    <span className="text-xs text-slate-500">Location</span>
                    <span className="text-sm text-slate-900">{zone.location}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-xs text-slate-500">Sensors</span>
                    <span className="text-sm text-slate-900">{zone.active_sensors}/{zone.total_sensors} Active</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-xs text-slate-500">Water Loss</span>
                    <span className={cn(
                      "text-sm font-medium",
                      zone.water_loss_percentage > 10 ? "text-amber-600" : "text-emerald-600"
                    )}>{zone.water_loss_percentage}%</span>
                  </div>
                </div>
                <Button className="w-full bg-slate-100 border border-slate-200 text-slate-700 hover:bg-slate-200 rounded-xl group-hover:border-slate-300">
                  Edit Zone
                  <ChevronRight className="w-4 h-4 ml-1" />
                </Button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Settings Tab */}
      {selectedTab === 'settings' && (
        <div className="space-y-6">
          <h2 className="text-xl font-semibold text-slate-900 flex items-center gap-2">
            <SettingsIcon className="w-5 h-5 text-blue-600" />
            System Settings
          </h2>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Alert Thresholds */}
            <div className="bg-white/80 backdrop-blur-sm rounded-2xl border border-slate-200 shadow-lg p-6">
              <h3 className="text-lg font-semibold text-slate-900 mb-4 flex items-center gap-2">
                <Shield className="w-5 h-5 text-amber-600" />
                Alert Thresholds
              </h3>
              <div className="space-y-4">
                <div>
                  <label className="text-sm font-medium text-slate-600">Pressure Drop Threshold (bar)</label>
                  <Input type="number" defaultValue="0.5" className="mt-2 bg-slate-50 border-slate-200 text-slate-900 rounded-xl focus:border-slate-400" />
                </div>
                <div>
                  <label className="text-sm font-medium text-slate-600">Flow Rate Anomaly (%)</label>
                  <Input type="number" defaultValue="15" className="mt-2 bg-slate-50 border-slate-200 text-slate-900 rounded-xl focus:border-slate-400" />
                </div>
                <div>
                  <label className="text-sm font-medium text-slate-600">Water Loss Alert (%)</label>
                  <Input type="number" defaultValue="10" className="mt-2 bg-slate-50 border-slate-200 text-slate-900 rounded-xl focus:border-slate-400" />
                </div>
                <Button className="w-full bg-linear-to-r from-slate-800 to-slate-900 text-white hover:opacity-90 rounded-xl">Save Thresholds</Button>
              </div>
            </div>

            {/* Notification Settings */}
            <div className="bg-white/80 backdrop-blur-sm rounded-2xl border border-slate-200 shadow-lg p-6">
              <h3 className="text-lg font-semibold text-slate-900 mb-4">Notification Settings</h3>
              <div className="space-y-4">
                <div className="flex items-center justify-between p-3 bg-slate-50 rounded-xl">
                  <label className="text-sm font-medium text-slate-900">Email Alerts</label>
                  <div className="relative">
                    <input type="checkbox" defaultChecked className="sr-only peer" id="email-toggle" />
                    <label htmlFor="email-toggle" className="w-11 h-6 bg-slate-200 rounded-full peer peer-checked:bg-linear-to-r peer-checked:from-slate-700 peer-checked:to-slate-900 cursor-pointer block transition-all duration-300 after:content-[''] after:absolute after:top-0.5 after:left-0.5 after:w-5 after:h-5 after:bg-white after:rounded-full after:transition-all peer-checked:after:translate-x-5 after:shadow-sm"></label>
                  </div>
                </div>
                <div className="flex items-center justify-between p-3 bg-slate-50 rounded-xl">
                  <label className="text-sm font-medium text-slate-900">SMS Alerts</label>
                  <div className="relative">
                    <input type="checkbox" className="sr-only peer" id="sms-toggle" />
                    <label htmlFor="sms-toggle" className="w-11 h-6 bg-slate-200 rounded-full peer peer-checked:bg-linear-to-r peer-checked:from-slate-700 peer-checked:to-slate-900 cursor-pointer block transition-all duration-300 after:content-[''] after:absolute after:top-0.5 after:left-0.5 after:w-5 after:h-5 after:bg-white after:rounded-full after:transition-all peer-checked:after:translate-x-5 after:shadow-sm"></label>
                  </div>
                </div>
                <div className="flex items-center justify-between p-3 bg-slate-50 rounded-xl">
                  <label className="text-sm font-medium text-slate-900">Daily Reports</label>
                  <div className="relative">
                    <input type="checkbox" defaultChecked className="sr-only peer" id="daily-toggle" />
                    <label htmlFor="daily-toggle" className="w-11 h-6 bg-slate-200 rounded-full peer peer-checked:bg-linear-to-r peer-checked:from-slate-700 peer-checked:to-slate-900 cursor-pointer block transition-all duration-300 after:content-[''] after:absolute after:top-0.5 after:left-0.5 after:w-5 after:h-5 after:bg-white after:rounded-full after:transition-all peer-checked:after:translate-x-5 after:shadow-sm"></label>
                  </div>
                </div>
                <div className="flex items-center justify-between p-3 bg-slate-50 rounded-xl">
                  <label className="text-sm font-medium text-slate-900">Weekly Summary</label>
                  <div className="relative">
                    <input type="checkbox" defaultChecked className="sr-only peer" id="weekly-toggle" />
                    <label htmlFor="weekly-toggle" className="w-11 h-6 bg-slate-200 rounded-full peer peer-checked:bg-linear-to-r peer-checked:from-slate-700 peer-checked:to-slate-900 cursor-pointer block transition-all duration-300 after:content-[''] after:absolute after:top-0.5 after:left-0.5 after:w-5 after:h-5 after:bg-white after:rounded-full after:transition-all peer-checked:after:translate-x-5 after:shadow-sm"></label>
                  </div>
                </div>
                <Button className="w-full bg-linear-to-r from-slate-800 to-slate-900 text-white hover:opacity-90 rounded-xl">Save Settings</Button>
              </div>
            </div>
          </div>

          {/* Backup & Maintenance */}
          <div className="bg-white/80 backdrop-blur-sm rounded-2xl border border-slate-200 shadow-lg p-6">
            <h3 className="text-lg font-semibold text-slate-900 mb-4 flex items-center gap-2">
              <Database className="w-5 h-5 text-emerald-600" />
              Backup & Maintenance
            </h3>
            <div className="space-y-4">
              <div className="flex items-center justify-between p-4 bg-slate-50 rounded-xl">
                <div>
                  <p className="font-medium text-slate-900">Last Backup</p>
                  <p className="text-sm text-slate-500">February 25, 2024 - 2:30 AM</p>
                </div>
                <Button className="bg-linear-to-r from-slate-800 to-slate-900 text-white hover:opacity-90 rounded-xl">
                  Backup Now
                </Button>
              </div>
              <div className="flex items-center justify-between p-4 bg-slate-50 rounded-xl">
                <div>
                  <p className="font-medium text-slate-900">Database Maintenance</p>
                  <p className="text-sm text-slate-500">Last run: February 20, 2024</p>
                </div>
                <Button className="bg-slate-100 border border-slate-200 text-slate-700 hover:bg-slate-200 rounded-xl">
                  Schedule
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
