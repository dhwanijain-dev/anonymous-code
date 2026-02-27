'use client';

import { useEffect, useRef, useCallback } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { Sensor } from '@/lib/types';

interface FullMapProps {
  sensors: Sensor[];
  selectedSensor: Sensor | null;
  onSensorSelect: (sensor: Sensor | null) => void;
  center?: [number, number];
  zoom?: number;
}

// Custom marker icons
const createCustomIcon = (sensor: Sensor, isSelected: boolean) => {
  const isAnomaly = sensor.last_reading?.is_anomaly;
  const baseColor = isAnomaly ? '#f59e0b' : '#10b981';
  const darkColor = isAnomaly ? '#d97706' : '#059669';
  const size = isSelected ? 32 : 24;
  
  return L.divIcon({
    className: 'custom-marker',
    html: `
      <div style="
        width: ${size}px;
        height: ${size}px;
        background: linear-gradient(135deg, ${baseColor}, ${darkColor});
        border-radius: 50% 50% 50% 0;
        transform: rotate(-45deg);
        border: ${isSelected ? '3px' : '2px'} solid white;
        box-shadow: 0 ${isSelected ? '4px 12px' : '2px 8px'} rgba(0,0,0,${isSelected ? '0.4' : '0.3'});
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.2s ease;
        ${isAnomaly ? 'animation: pulse 1.5s infinite;' : ''}
      ">
        <div style="
          width: ${isSelected ? '10px' : '8px'};
          height: ${isSelected ? '10px' : '8px'};
          background: white;
          border-radius: 50%;
          transform: rotate(45deg);
        "></div>
      </div>
    `,
    iconSize: [size, size],
    iconAnchor: [size / 2, size],
    popupAnchor: [0, -size],
  });
};

// Get sensor type icon SVG
const getSensorIconSvg = (type: Sensor['sensor_type']) => {
  switch (type) {
    case 'flow':
      return '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/>';
    case 'pressure':
      return '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4"/>';
    case 'leak_detection':
      return '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/>';
    case 'acoustic':
      return '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5.636 18.364a9 9 0 010-12.728m12.728 0a9 9 0 010 12.728m-9.9-2.829a5 5 0 010-7.07m7.072 0a5 5 0 010 7.07M13 12a1 1 0 11-2 0 1 1 0 012 0z"/>';
    default:
      return '<circle cx="12" cy="12" r="3"/>';
  }
};

export default function FullLeafletMap({ 
  sensors, 
  selectedSensor,
  onSensorSelect,
  center = [40.7128, -74.006], 
  zoom = 14 
}: FullMapProps) {
  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  const markersRef = useRef<Map<string, L.Marker>>(new Map());

  // Update marker icons when selection changes
  const updateMarkerIcons = useCallback(() => {
    sensors.forEach((sensor) => {
      const marker = markersRef.current.get(sensor.id);
      if (marker) {
        marker.setIcon(createCustomIcon(sensor, selectedSensor?.id === sensor.id));
      }
    });
  }, [sensors, selectedSensor]);

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
      const marker = L.marker([sensor.coordinates.latitude, sensor.coordinates.longitude], {
        icon: createCustomIcon(sensor, false),
      }).addTo(map);

      markersRef.current.set(sensor.id, marker);

      const isAnomaly = sensor.last_reading?.is_anomaly;
      const statusBg = isAnomaly ? '#fef3c7' : '#d1fae5';
      const statusColor = isAnomaly ? '#d97706' : '#059669';
      const statusText = isAnomaly ? 'Anomaly' : 'Normal';

      // Add popup
      marker.bindPopup(`
        <div style="
          font-family: system-ui, -apple-system, sans-serif;
          padding: 8px;
          min-width: 180px;
        ">
          <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
            <div style="
              width: 32px;
              height: 32px;
              background: linear-gradient(135deg, #3b82f6, #1d4ed8);
              border-radius: 8px;
              display: flex;
              align-items: center;
              justify-content: center;
            ">
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2">
                ${getSensorIconSvg(sensor.sensor_type)}
              </svg>
            </div>
            <div>
              <div style="font-weight: 600; color: #0f172a; font-size: 14px;">${sensor.name}</div>
              <div style="font-size: 11px; color: #64748b; text-transform: uppercase;">${sensor.zone_id}</div>
            </div>
          </div>
          
          <div style="display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 8px;">
            <span style="
              display: inline-flex;
              align-items: center;
              gap: 4px;
              padding: 3px 8px;
              border-radius: 9999px;
              font-size: 11px;
              font-weight: 500;
              background: ${statusBg};
              color: ${statusColor};
            ">
              <span style="width: 6px; height: 6px; border-radius: 50%; background: currentColor;"></span>
              ${statusText}
            </span>
            <span style="
              display: inline-flex;
              align-items: center;
              gap: 4px;
              padding: 3px 8px;
              border-radius: 9999px;
              font-size: 11px;
              font-weight: 500;
              background: #f1f5f9;
              color: #475569;
            ">
              ${sensor.sensor_type.replace('_', ' ')}
            </span>
          </div>
          
          <div style="
            background: #f8fafc;
            border-radius: 8px;
            padding: 8px;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
            font-size: 11px;
          ">
            <div>
              <div style="color: #64748b;">Reading</div>
              <div style="font-weight: 600; color: #0f172a;">${sensor.last_reading?.value} ${sensor.last_reading?.unit}</div>
            </div>
            <div>
              <div style="color: #64748b;">Battery</div>
              <div style="font-weight: 600; color: ${(sensor.battery_level ?? 0) > 50 ? '#059669' : (sensor.battery_level ?? 0) > 20 ? '#d97706' : '#dc2626'};">${sensor.battery_level ?? 0}%</div>
            </div>
          </div>
        </div>
      `, {
        closeButton: false,
        className: 'custom-popup',
        maxWidth: 250,
      });

      // Handle click
      marker.on('click', () => {
        onSensorSelect(sensor);
      });
    });

    mapInstanceRef.current = map;

    // Cleanup
    return () => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove();
        mapInstanceRef.current = null;
        markersRef.current.clear();
      }
    };
  }, [sensors, center, zoom, onSensorSelect]);

  // Update icons when selection changes
  useEffect(() => {
    updateMarkerIcons();
    
    // Pan to selected sensor
    if (selectedSensor && mapInstanceRef.current) {
      mapInstanceRef.current.panTo([
        selectedSensor.coordinates.latitude,
        selectedSensor.coordinates.longitude
      ], { animate: true, duration: 0.5 });
    }
  }, [selectedSensor, updateMarkerIcons]);

  return (
    <>
      <style jsx global>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; transform: rotate(-45deg) scale(1); }
          50% { opacity: 0.8; transform: rotate(-45deg) scale(1.1); }
        }
        .custom-marker {
          background: transparent;
          border: none;
        }
        .custom-popup .leaflet-popup-content-wrapper {
          background: white;
          border-radius: 12px;
          box-shadow: 0 4px 20px rgba(0,0,0,0.15);
          border: 1px solid #e2e8f0;
          padding: 0;
        }
        .custom-popup .leaflet-popup-content {
          margin: 0;
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
          border-bottom: 1px solid #e2e8f0 !important;
          width: 32px !important;
          height: 32px !important;
          line-height: 32px !important;
          font-size: 16px !important;
        }
        .leaflet-control-zoom a:last-child {
          border-bottom: none !important;
        }
        .leaflet-control-zoom a:hover {
          background: #f1f5f9 !important;
        }
        .leaflet-control-attribution {
          background: rgba(255,255,255,0.9) !important;
          padding: 4px 8px !important;
          font-size: 10px !important;
          border-radius: 6px !important;
          margin: 8px !important;
        }
        .leaflet-control-attribution a {
          color: #64748b !important;
        }
      `}</style>
      <div 
        ref={mapRef} 
        className="w-full h-full"
        style={{ minHeight: '100%' }}
      />
    </>
  );
}
