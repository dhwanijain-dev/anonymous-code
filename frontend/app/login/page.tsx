'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { AlertCircle, Droplet, Waves, Mail, Lock, ArrowRight } from 'lucide-react';
import { login, setAuthToken } from '@/lib/auth';
import { useToast } from '@/hooks/use-toast';

export default function LoginPage() {
  const router = useRouter();
  const { toast } = useToast();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);

    try {
      const response = await login(email, password);
      setAuthToken(response.token);
      toast({
        title: 'Welcome back!',
        description: `Logged in as ${response.user.name}`,
      });
      router.push('/dashboard');
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Login failed. Please try again.';
      setError(message);
      toast({
        title: 'Login failed',
        description: message,
        variant: 'destructive',
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen relative overflow-hidden flex items-center justify-center p-4">
      {/* Light gradient background */}
      <div className="absolute inset-0 bg-linear-to-br from-slate-50 via-blue-50 to-indigo-100" />

      {/* Subtle decorative orbs */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute -top-40 -right-40 w-[500px] h-[500px] bg-blue-200/40 rounded-full blur-[120px]" />
        <div className="absolute -bottom-40 -left-40 w-[600px] h-[600px] bg-indigo-200/40 rounded-full blur-[120px]" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[700px] h-[700px] bg-purple-100/30 rounded-full blur-[140px]" />
      </div>

      {/* Subtle wave pattern overlay */}
      <div className="absolute inset-0 opacity-20">
        <svg className="absolute bottom-0 w-full" viewBox="0 0 1440 320" preserveAspectRatio="none">
          <path fill="#1e293b" fillOpacity="0.1" d="M0,96L48,112C96,128,192,160,288,160C384,160,480,128,576,122.7C672,117,768,139,864,154.7C960,171,1056,181,1152,165.3C1248,149,1344,107,1392,85.3L1440,64L1440,320L1392,320C1344,320,1248,320,1152,320C1056,320,960,320,864,320C768,320,672,320,576,320C480,320,384,320,288,320C192,320,96,320,48,320L0,320Z"></path>
        </svg>
      </div>

      <div className="relative z-10 w-full max-w-md">
        {/* Logo */}
        <div className="flex justify-center mb-8">
          <div className="flex items-center gap-3 text-slate-900">
            <div className="relative">
              <div className="absolute inset-0 bg-blue-500/20 blur-xl rounded-full" />
              <div className="relative bg-linear-to-br from-slate-800 to-slate-900 p-3 rounded-2xl shadow-xl">
                <Droplet className="w-8 h-8 fill-white text-white" />
              </div>
            </div>
            <div>
              <span className="text-3xl font-bold tracking-tight">Water Detection System</span>
              <div className="flex items-center gap-1 text-slate-500 text-xs">
                <Waves className="w-3 h-3" />
                <span>Smart Water Monitoring</span>
              </div>
            </div>
          </div>
        </div>

        {/* Card */}
        <div className="bg-white/80 backdrop-blur-xl rounded-3xl border border-slate-200 shadow-2xl shadow-slate-200/50 p-8">
          <div className="text-center mb-8">
            <h1 className="text-3xl font-bold text-slate-900 mb-2">Welcome Back</h1>
            <p className="text-slate-500">Sign in to monitor your water systems</p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-5">
            {error && (
              <div className="flex gap-3 p-4 bg-red-50 text-red-700 rounded-xl border border-red-200">
                <AlertCircle className="w-5 h-5 mt-0.5 shrink-0" />
                <p className="text-sm">{error}</p>
              </div>
            )}

            <div className="space-y-2">
              <Label htmlFor="email" className="text-slate-700 font-medium">Email</Label>
              <div className="relative">
                <Mail className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
                <Input
                  id="email"
                  type="email"
                  placeholder="admin@example.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                  disabled={isLoading}
                  className="pl-12 h-12 bg-slate-50 border-slate-200 text-slate-900 placeholder:text-slate-400 rounded-xl focus:bg-white focus:border-slate-400 transition-all duration-300"
                />
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="password" className="text-slate-700 font-medium">Password</Label>
              <div className="relative">
                <Lock className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
                <Input
                  id="password"
                  type="password"
                  placeholder="••••••••"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  disabled={isLoading}
                  className="pl-12 h-12 bg-slate-50 border-slate-200 text-slate-900 placeholder:text-slate-400 rounded-xl focus:bg-white focus:border-slate-400 transition-all duration-300"
                />
              </div>
            </div>

            <Button
              type="submit"
              className="w-full h-12 bg-linear-to-r from-slate-800 to-slate-900 hover:from-slate-900 hover:to-black text-white rounded-xl font-semibold text-base transition-all duration-300 group shadow-lg shadow-slate-300"
              disabled={isLoading}
            >
              {isLoading ? (
                <span className="flex items-center gap-2">
                  <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  Signing in...
                </span>
              ) : (
                <span className="flex items-center gap-2">
                  Sign In
                  <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                </span>
              )}
            </Button>

            <div className="text-center text-sm pt-2">
              <span className="text-slate-500">Don't have an account? </span>
              <Link href="/register" className="text-slate-900 font-semibold hover:text-slate-700 transition-colors">
                Register
              </Link>
            </div>
          </form>
        </div>

        {/* Demo credentials */}
        <div className="mt-6 bg-white/60 backdrop-blur-lg p-4 rounded-2xl border border-slate-200 shadow-lg">
          <p className="text-xs font-semibold text-slate-600 mb-2 flex items-center gap-2">
            <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse" />
            Demo Credentials
          </p>
          <div className="flex justify-between text-xs text-slate-500">
            <span>Email: admin@aquaguard.com</span>
            <span>Pass: demo123456</span>
          </div>
        </div>

        {/* DEV ONLY: Dashboard bypass button */}
        {process.env.NODE_ENV === 'development' && (
          <Button
            type="button"
            variant="outline"
            onClick={() => {
              const devToken = btoa(JSON.stringify({ email: 'admin@aquaguard.com', timestamp: Date.now() }));
              setAuthToken(devToken);
              router.push('/dashboard');
            }}
            className="mt-4 w-full h-10 bg-amber-50 border-amber-300 text-amber-700 hover:bg-amber-100 rounded-xl text-sm font-medium"
          >
            ⚡ DEV: Skip to Dashboard
          </Button>
        )}
      </div>
    </div>
  );
}
