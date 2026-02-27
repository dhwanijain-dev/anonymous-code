'use client';

import { useEffect, useRef } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

interface Sensor {
  id: number;
  name: string;
  lat: number;
  lng: number;
  status: 'active' | 'warning' | 'error';
}

interface LeafletMapProps {
  sensors: Sensor[];
  center?: [number, number];
  zoom?: number;
}

// Custom marker icons
const createCustomIcon = (status: string) => {
  const color = status === 'active' ? '#10b981' : status === 'warning' ? '#f59e0b' : '#ef4444';
  const shadowColor = status === 'active' ? '#059669' : status === 'warning' ? '#d97706' : '#dc2626';
  
  return L.divIcon({
    className: 'custom-marker',
    html: `
      <div style="
        width: 24px;
        height: 24px;
        background: linear-gradient(135deg, ${color}, ${shadowColor});
        border-radius: 50% 50% 50% 0;
        transform: rotate(-45deg);
        border: 2px solid white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        display: flex;
        align-items: center;
        justify-content: center;
      ">
        <div style="
          width: 8px;
          height: 8px;
          background: white;
          border-radius: 50%;
          transform: rotate(45deg);
        "></div>
      </div>
    `,
    iconSize: [24, 24],
    iconAnchor: [12, 24],
    popupAnchor: [0, -24],
  });
};

export default function LeafletMap({ 
  sensors, 
  center = [40.7128, -74.006], 
  zoom = 14 
}: LeafletMapProps) {
  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);

  useEffect(() => {
    if (!mapRef.current || mapInstanceRef.current) return;

    // Initialize map
    const map = L.map(mapRef.current, {
      center: center,
      zoom: zoom,
      zoomControl: false,
      attributionControl: false,
    });

    // Add tile layer with light theme
    L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
      maxZoom: 19,
    }).addTo(map);

    // Add zoom control to bottom right
    L.control.zoom({ position: 'bottomright' }).addTo(map);

    // Add attribution
    L.control.attribution({ position: 'bottomleft', prefix: false })
      .addAttribution('© <a href="https://carto.com/">CARTO</a>')
      .addTo(map);

    // Add markers for each sensor
    sensors.forEach((sensor) => {
      const marker = L.marker([sensor.lat, sensor.lng], {
        icon: createCustomIcon(sensor.status),
      }).addTo(map);

      // Add popup
      marker.bindPopup(`
        <div style="
          font-family: system-ui, -apple-system, sans-serif;
          padding: 4px;
          min-width: 120px;
        ">
          <div style="font-weight: 600; color: #0f172a; margin-bottom: 4px;">${sensor.name}</div>
          <div style="
            display: inline-flex;
            align-items: center;
            gap: 4px;
            padding: 2px 8px;
            border-radius: 9999px;
            font-size: 11px;
            font-weight: 500;
            background: ${sensor.status === 'active' ? '#d1fae5' : sensor.status === 'warning' ? '#fef3c7' : '#fee2e2'};
            color: ${sensor.status === 'active' ? '#059669' : sensor.status === 'warning' ? '#d97706' : '#dc2626'};
          ">
            <span style="
              width: 6px;
              height: 6px;
              border-radius: 50%;
              background: currentColor;
            "></span>
            ${sensor.status.charAt(0).toUpperCase() + sensor.status.slice(1)}
          </div>
        </div>
      `, {
        closeButton: false,
        className: 'custom-popup',
      });
    });

    mapInstanceRef.current = map;

    // Cleanup
    return () => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove();
        mapInstanceRef.current = null;
      }
    };
  }, [sensors, center, zoom]);

  return (
    <>
      <style jsx global>{`
        .custom-marker {
          background: transparent;
          border: none;
        }
        .custom-popup .leaflet-popup-content-wrapper {
          background: white;
          border-radius: 12px;
          box-shadow: 0 4px 20px rgba(0,0,0,0.15);
          border: 1px solid #e2e8f0;
        }
        .custom-popup .leaflet-popup-tip {
          background: white;
          border: 1px solid #e2e8f0;
          box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        }
        .leaflet-control-zoom {
          border: none !important;
          box-shadow: 0 2px 10px rgba(0,0,0,0.1) !important;
          border-radius: 8px !important;
          overflow: hidden;
        }
        .leaflet-control-zoom a {
          background: white !important;
          color: #334155 !important;
          border: none !important;
          width: 28px !important;
          height: 28px !important;
          line-height: 28px !important;
          font-size: 14px !important;
        }
        .leaflet-control-zoom a:hover {
          background: #f1f5f9 !important;
        }
        .leaflet-control-attribution {
          background: rgba(255,255,255,0.8) !important;
          padding: 2px 6px !important;
          font-size: 10px !important;
          border-radius: 4px !important;
        }
        .leaflet-control-attribution a {
          color: #64748b !important;
        }
      `}</style>
      <div 
        ref={mapRef} 
        className="w-full h-full rounded-xl"
        style={{ minHeight: '100%' }}
      />
    </>
  );
}
