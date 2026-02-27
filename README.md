
#Urban Water Leakage Detection System

> **Hackathon Project** | Real-time IoT-powered water leak detection with GIS visualization, ML anomaly detection, and instant operator alerts.


---

## 🚨 The Problem

Up to **30–40% of urban water supply** is lost to undetected pipe leaks. Manual detection is slow, expensive, and purely reactive. AquaAlert changes that with real-time, intelligent monitoring.

---

## ✅ What We Built

A full-stack IoT platform that:
- Reads **flow, pressure, vibration, and temperature** from ESP32 sensor nodes installed at pipe junctions
- Streams data over **MQTT** to a cloud backend for storage and analysis
- Runs **dual-stage leak detection** — rule-based (real-time) + ML-based (Isolation Forest)
- Visualises live sensor status, leak alerts, and risk zones on an **interactive GIS map**
- Notifies operators instantly and supports full **alert acknowledgement + resolution workflows**

---

## 🏗️ Architecture

```
[ESP32 Sensor Nodes]
        │  MQTT / WiFi
        ▼
[Raspberry Pi Gateway]  ──── Edge filtering & buffering
        │  MQTT / 4G fallback
        ▼
[AWS IoT Core / Mosquitto]
        │
        ▼
[Node.js API + FastAPI Analytics]
        │
   ┌────┴────┐
   ▼         ▼
[TimescaleDB] [Redis]
        │
        ▼
[React Dashboard + Leaflet Map]
```

| Layer | Component | Technology |
|-------|-----------|------------|
| Edge | Sensor Node | ESP32 + YF-S201 + MPX5700AP + SW-420 |
| Edge | Gateway | Raspberry Pi 4 |
| Transport | Protocol | MQTT over WiFi / 4G |
| Cloud | API Server | Node.js / FastAPI |
| Cloud | Database | PostgreSQL 15 + TimescaleDB |
| Cloud | Cache | Redis 7 |
| Cloud | ML | Python + Scikit-learn (Isolation Forest) |
| Frontend | Web App | React 18 + Leaflet.js + Recharts |

---

## 🔧 Hardware (Per Sensor Node, ~₹2,050)

| Component | Model | Purpose |
|-----------|-------|---------|
| Microcontroller | ESP32 DevKit v1 | WiFi + processing |
| Flow Sensor | YF-S201 | Water flow rate (L/min) |
| Pressure Sensor | MPX5700AP | Pipe pressure (0–700 kPa) |
| Vibration Sensor | SW-420 | Leak vibration detection |
| Temperature Sensor | DS18B20 | Ambient + water temp |
| Enclosure | IP67 Junction Box | Waterproof housing |

**Sampling rates:** Flow every 5s · Pressure every 2s · Vibration event-driven · Temp every 30s

---

## 🧠 Leak Detection

### Stage 1 — Rule-Based (< 100ms, real-time)
- Pressure drop > 15% within 30 seconds → **High alert**
- Flow rate at dead-end node > 2 L/min → **Leak suspected**
- Vibration + pressure drop together → **Critical alert**
- Node silent > 60 seconds → **Sensor fault**

### Stage 2 — ML-Based (every 5 minutes)
- **Isolation Forest** trained on historical flow/pressure patterns
- Features: `flow_rate`, `pressure`, `hour_of_day`, `day_of_week`, `rolling_mean_1h`, `rolling_std_1h`
- Target false positive rate: **< 2%**
- Model retrained weekly with labelled data from resolved alerts

---

## 🖥️ Dashboard Pages

| Page | Route | Purpose |
|------|-------|---------|
| Dashboard | `/dashboard` | KPI overview, live alerts, mini map |
| Live Map | `/map` | Full GIS map with real-time node status |
| Sensor Detail | `/sensors/:id` | Charts, readings, anomaly overlays |
| Alerts | `/alerts` | Alert log with acknowledge/resolve workflow |
| Analytics | `/analytics` | Water loss estimation, risk scores |
| Settings | `/settings` | Thresholds, notification channels |
| Admin | `/admin/*` | User management, system health |

---

## 🚀 Quick Start

### Prerequisites
- Node.js v20+, Python 3.11+, Docker Desktop
- PostgreSQL 15 with TimescaleDB extension
- Redis 7, Mosquitto MQTT broker

### 1. Clone & Configure
```bash
git clone https://github.com/your-team/aquaalert.git
cd aquaalert
cp .env.example .env
# Edit .env with your credentials
```

### 2. Environment Variables
```env
DATABASE_URL=postgresql://user:pass@localhost:5432/aquaalert
REDIS_URL=redis://localhost:6379
MQTT_BROKER=mqtt://localhost:1883
JWT_SECRET=<strong-random-secret>
JWT_EXPIRES_IN=15m
REFRESH_TOKEN_EXPIRES=7d
FRONTEND_URL=http://localhost:5173
```

### 3. Start Everything
```bash
# Option A: Docker (recommended)
docker-compose up

# Option B: Manual
cd backend && npm install && npm run dev
cd frontend && npm install && npm run dev
cd analytics && pip install -r requirements.txt && uvicorn main:app --reload
```

### 4. Open Dashboard
Visit `http://localhost:5173` and log in with demo credentials below.

---

## 🔑 Demo Credentials

| Role | Email | Password | Access |
|------|-------|----------|--------|
| Admin | `admin@aquaalert.io` | `Admin@1234` | Full system access |
| Operator | `ops@aquaalert.io` | `Ops@1234` | View + alert management |
| Viewer | `viewer@aquaalert.io` | `View@1234` | Read-only dashboard |

---

## 🎬 Demo Flow (For Evaluators)

Follow these steps for maximum impact:

1. **Login** as Operator → see role-based auth
2. **Dashboard** → KPI cards, live alert feed, mini map
3. **Live Map** → colour-coded sensor nodes across zones
4. **Simulate a leak** (see below) → node turns red in real-time
5. **Alert Detail** → timeline, severity, affected node info
6. **Sensor Detail** → pressure drop chart with anomaly shading
7. **Acknowledge + Resolve** → full operator workflow
8. **Analytics** → water loss estimation and ML risk scores
9. **Admin → Users** → role-based access control
10. **System Health** → DB, MQTT, Redis status

### Simulating a Leak (No Hardware Needed)
```bash
# Option A: Seed script
node scripts/simulate_leak.js --node <nodeId> --severity high

# Option B: Direct API call
POST /api/readings/ingest
{
  "nodeId": "<nodeId>",
  "pressure": 120,
  "flow_rate": 0.2,
  "vibration": true,
  "temperature": 24.5
}
```
This triggers the rule-based alert immediately and updates the live map via WebSocket.

---

## 📡 Key API Endpoints

```
POST   /api/auth/login                  # Login
GET    /api/nodes                       # List all sensor nodes
GET    /api/nodes/:id/readings          # Historical readings
GET    /api/alerts                      # List alerts (filterable)
POST   /api/alerts/:id/acknowledge      # Acknowledge alert
GET    /api/analytics/water-loss        # Water loss by zone
GET    /api/analytics/predictions       # ML risk scores per node
GET    /api/admin/system/health         # System health check
```

**WebSocket events (Socket.io):** `reading:new` · `alert:created` · `alert:updated` · `node:status`

---

## 🔒 Security

- JWT access tokens (15 min) + httpOnly refresh token cookies
- bcrypt password hashing (salt rounds = 12)
- MQTT over TLS 1.2 with client certificates
- API rate limiting: 100 req / 15 min per IP
- Zod schema validation on all inputs
- Parameterised queries via Prisma ORM
- Role-based access control on every protected route

---

## 🌟 Key Differentiators

| Feature | Why It Matters |
|---------|----------------|
| Real-time WebSocket updates | Live data changes visible to judges — not a static mockup |
| GIS Map with live node status | Visual impact, clear geographic relevance |
| Rule + ML dual detection | Engineering depth beyond simple threshold alerts |
| Water loss quantification | Translates tech into real-world monetary/resource impact |
| Simulation script | Reliable demo regardless of hardware availability |
| TimescaleDB time-series | Production-scale data architecture |
| Role-based access control | Enterprise-ready, security-conscious design |
| Modular microservice design | Scalable and maintainable architecture |

---

