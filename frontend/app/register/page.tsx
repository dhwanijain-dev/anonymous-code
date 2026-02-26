'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { AlertCircle, Droplet, Waves, Mail, Lock, User, ArrowRight, CheckCircle2 } from 'lucide-react';
import { register, setAuthToken } from '@/lib/auth';
import { useToast } from '@/hooks/use-toast';

export default function RegisterPage() {
  const router = useRouter();
  const { toast } = useToast();
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    if (password !== confirmPassword) {
      setError('Passwords do not match');
      return;
    }

    if (password.length < 8) {
      setError('Password must be at least 8 characters');
      return;
    }

    setIsLoading(true);

    try {
      const response = await register(email, password, name);
      setAuthToken(response.token);
      toast({
        title: 'Account created!',
        description: 'Welcome to AquaGuard',
      });
      router.push('/dashboard');
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Registration failed. Please try again.';
      setError(message);
      toast({
        title: 'Registration failed',
        description: message,
        variant: 'destructive',
      });
    } finally {
      setIsLoading(false);
    }
  };

  const passwordRequirements = [
    { met: password.length >= 8, text: 'At least 8 characters' },
    { met: password === confirmPassword && password.length > 0, text: 'Passwords match' },
  ];

  return (
    <div className="min-h-screen relative overflow-hidden flex items-center justify-center p-4">
      {/* Light gradient background */}
      <div className="absolute inset-0 bg-linear-to-br from-slate-50 via-blue-50 to-indigo-100" />

      {/* Subtle floating orbs */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute -top-40 -left-40 w-[500px] h-[500px] bg-blue-200/40 rounded-full blur-[120px] animate-pulse" />
        <div className="absolute -bottom-40 -right-40 w-[600px] h-[600px] bg-indigo-200/40 rounded-full blur-[120px] animate-pulse" style={{ animationDelay: '1s' }} />
        <div className="absolute top-1/4 right-1/4 w-[400px] h-[400px] bg-cyan-200/30 rounded-full blur-[100px] animate-pulse" style={{ animationDelay: '0.7s' }} />
        <div className="absolute bottom-1/3 left-1/3 w-[450px] h-[450px] bg-slate-200/30 rounded-full blur-[120px] animate-pulse" style={{ animationDelay: '1.3s' }} />
      </div>

      {/* Decorative water bubbles */}
      <div className="absolute inset-0 pointer-events-none">
        {[...Array(6)].map((_, i) => (
          <div
            key={i}
            className="absolute w-4 h-4 bg-slate-300/40 rounded-full animate-bounce"
            style={{
              left: `${15 + i * 15}%`,
              bottom: `${10 + (i % 3) * 20}%`,
              animationDelay: `${i * 0.3}s`,
              animationDuration: `${2 + i * 0.5}s`,
            }}
          />
        ))}
      </div>

      <div className="relative z-10 w-full max-w-md">
        {/* Logo */}
        <div className="flex justify-center mb-6">
          <div className="flex items-center gap-3 text-slate-900">
            <div className="relative">
              <div className="absolute inset-0 bg-slate-300/50 blur-xl rounded-full" />
              <div className="relative bg-linear-to-br from-slate-800 to-slate-900 p-3 rounded-2xl border border-slate-300 shadow-lg">
                <Droplet className="w-8 h-8 fill-white text-white" />
              </div>
            </div>
            <div>
              <span className="text-3xl font-bold tracking-tight">AquaGuard</span>
              <div className="flex items-center gap-1 text-slate-500 text-xs">
                <Waves className="w-3 h-3" />
                <span>Smart Water Monitoring</span>
              </div>
            </div>
          </div>
        </div>

        {/* Light glassmorphism card */}
        <div className="backdrop-blur-xl bg-white/80 rounded-3xl border border-slate-200 shadow-2xl p-8">
          <div className="text-center mb-6">
            <h1 className="text-3xl font-bold text-slate-900 mb-2">Create Account</h1>
            <p className="text-slate-500">Start monitoring water leakage today</p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            {error && (
              <div className="flex gap-3 p-4 bg-red-50 text-red-700 rounded-xl border border-red-200">
                <AlertCircle className="w-5 h-5 mt-0.5 flex-shrink-0" />
                <p className="text-sm">{error}</p>
              </div>
            )}

            <div className="space-y-2">
              <Label htmlFor="name" className="text-slate-700 font-medium">Full Name</Label>
              <div className="relative">
                <User className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
                <Input
                  id="name"
                  type="text"
                  placeholder="John Doe"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  required
                  disabled={isLoading}
                  className="pl-12 h-12 bg-slate-50 border-slate-200 text-slate-900 placeholder:text-slate-400 rounded-xl focus:bg-white focus:border-slate-400 focus:ring-slate-400 transition-all duration-300"
                />
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="email" className="text-slate-700 font-medium">Email</Label>
              <div className="relative">
                <Mail className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
                <Input
                  id="email"
                  type="email"
                  placeholder="your@email.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                  disabled={isLoading}
                  className="pl-12 h-12 bg-slate-50 border-slate-200 text-slate-900 placeholder:text-slate-400 rounded-xl focus:bg-white focus:border-slate-400 focus:ring-slate-400 transition-all duration-300"
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
                  className="pl-12 h-12 bg-slate-50 border-slate-200 text-slate-900 placeholder:text-slate-400 rounded-xl focus:bg-white focus:border-slate-400 focus:ring-slate-400 transition-all duration-300"
                />
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="confirm-password" className="text-slate-700 font-medium">Confirm Password</Label>
              <div className="relative">
                <Lock className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
                <Input
                  id="confirm-password"
                  type="password"
                  placeholder="••••••••"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  required
                  disabled={isLoading}
                  className="pl-12 h-12 bg-slate-50 border-slate-200 text-slate-900 placeholder:text-slate-400 rounded-xl focus:bg-white focus:border-slate-400 focus:ring-slate-400 transition-all duration-300"
                />
              </div>
            </div>

            {/* Password requirements indicator */}
            {password.length > 0 && (
              <div className="space-y-2 p-3 bg-slate-50 rounded-xl border border-slate-200">
                {passwordRequirements.map((req, i) => (
                  <div key={i} className="flex items-center gap-2 text-xs">
                    <CheckCircle2 className={`w-4 h-4 transition-colors ${req.met ? 'text-emerald-500' : 'text-slate-300'}`} />
                    <span className={`transition-colors ${req.met ? 'text-emerald-600' : 'text-slate-400'}`}>{req.text}</span>
                  </div>
                ))}
              </div>
            )}

            <Button
              type="submit"
              className="w-full h-12 bg-linear-to-r from-slate-800 to-slate-900 text-white hover:from-slate-700 hover:to-slate-800 rounded-xl font-semibold text-base transition-all duration-300 group mt-2 shadow-lg"
              disabled={isLoading}
            >
              {isLoading ? (
                <span className="flex items-center gap-2">
                  <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  Creating account...
                </span>
              ) : (
                <span className="flex items-center gap-2">
                  Create Account
                  <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                </span>
              )}
            </Button>

            <div className="text-center text-sm pt-2">
              <span className="text-slate-500">Already have an account? </span>
              <Link href="/login" className="text-slate-900 font-semibold hover:text-slate-700 transition-colors">
                Sign In
              </Link>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}
