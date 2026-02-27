import { User } from './types';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3000/api';

export interface AuthResponse {
  token: string;
  user: User;
}

// Mock users for demo
const MOCK_USERS: Record<string, { password: string; user: User }> = {
  'admin@aquaguard.com': {
    password: 'demo123456',
    user: {
      id: '1',
      email: 'admin@aquaguard.com',
      name: 'Admin User',
      role: 'admin',
      created_at: new Date().toISOString(),
    },
  },
  'tech@aquaguard.com': {
    password: 'demo123456',
    user: {
      id: '2',
      email: 'tech@aquaguard.com',
      name: 'Technician User',
      role: 'technician',
      created_at: new Date().toISOString(),
    },
  },
};

export async function login(email: string, password: string): Promise<AuthResponse> {
  // For demo: Use mock authentication
  const mockUser = MOCK_USERS[email];
  if (mockUser && mockUser.password === password) {
    const token = btoa(JSON.stringify({ email, timestamp: Date.now() }));
    return {
      token,
      user: mockUser.user,
    };
  }

  // Try real API if configured
  if (process.env.NEXT_PUBLIC_API_URL) {
    const response = await fetch(`${API_BASE}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
      credentials: 'include',
    });

    if (!response.ok) {
      throw new Error('Login failed');
    }

    return response.json();
  }

  throw new Error('Invalid email or password');
}

export async function register(
  email: string,
  password: string,
  name: string
): Promise<AuthResponse> {
  // For demo: Mock registration
  if (!email || !password || !name) {
    throw new Error('All fields are required');
  }

  if (email in MOCK_USERS) {
    throw new Error('Email already exists');
  }

  // Create new mock user
  const newUser: User = {
    id: Date.now().toString(),
    email,
    name,
    role: 'technician',
    created_at: new Date().toISOString(),
  };

  const token = btoa(JSON.stringify({ email, timestamp: Date.now() }));
  return { token, user: newUser };
}

export async function logout(): Promise<void> {
  clearAuthToken();
  if (process.env.NEXT_PUBLIC_API_URL) {
    try {
      await fetch(`${API_BASE}/auth/logout`, {
        method: 'POST',
        credentials: 'include',
      });
    } catch {
      // Ignore errors from API logout
    }
  }
}

export async function getCurrentUser(): Promise<User | null> {
  try {
    const token = getAuthToken();
    if (!token) return null;

    // For demo: Decode mock token
    try {
      const decoded = JSON.parse(atob(token));
      const mockUser = MOCK_USERS[decoded.email];
      if (mockUser) {
        return mockUser.user;
      }
    } catch {
      // Continue to API call
    }

    // Try real API if configured
    if (process.env.NEXT_PUBLIC_API_URL) {
      const response = await fetch(`${API_BASE}/auth/me`, {
        credentials: 'include',
      });

      if (!response.ok) return null;
      return response.json();
    }

    return null;
  } catch {
    return null;
  }
}

export async function refreshToken(): Promise<string | null> {
  try {
    const currentToken = getAuthToken();
    if (!currentToken) return null;

    // For demo: Refresh mock token
    try {
      const decoded = JSON.parse(atob(currentToken));
      const newToken = btoa(JSON.stringify({ ...decoded, timestamp: Date.now() }));
      setAuthToken(newToken);
      return newToken;
    } catch {
      // Continue to API call
    }

    // Try real API if configured
    if (process.env.NEXT_PUBLIC_API_URL) {
      const response = await fetch(`${API_BASE}/auth/refresh`, {
        method: 'POST',
        credentials: 'include',
      });

      if (!response.ok) return null;
      const { token } = await response.json();
      return token;
    }

    return null;
  } catch {
    return null;
  }
}

export function setAuthToken(token: string): void {
  if (typeof window !== 'undefined') {
    localStorage.setItem('auth_token', token);
  }
}

export function getAuthToken(): string | null {
  if (typeof window !== 'undefined') {
    return localStorage.getItem('auth_token');
  }
  return null;
}

export function clearAuthToken(): void {
  if (typeof window !== 'undefined') {
    localStorage.removeItem('auth_token');
  }
}
