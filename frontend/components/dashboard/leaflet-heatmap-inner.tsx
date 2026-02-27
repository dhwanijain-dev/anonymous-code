'use client';

import { useEffect, useRef } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
// @ts-ignore - heatmap.js doesn't have types
import h337 from 'heatmap.js';

export interface HeatmapPoint {
  lat: number;
  lng: number;
  intensity: number;
  node: string;
  isRisk: boolean;
}

interface LeafletHeatmapInnerProps {
  points: HeatmapPoint[];
  minPressure: number;
  maxPressure: number;
}

// HeatmapOverlay class - adapted from leaflet-heatmap.js plugin
class HeatmapOverlay extends L.Layer {
  private cfg: any;
  private _el: HTMLDivElement;
  private _data: any[] = [];
  private _max: number = 1;
  private _min: number = 0;
  private _width: number = 0;
  private _height: number = 0;
  private _heatmap: any = null;
  private _origin: L.LatLng | null = null;

  constructor(config: any) {
    super();
    this.cfg = config;
    this._el = L.DomUtil.create('div', 'leaflet-zoom-hide') as HTMLDivElement;
    this._data = [];
    this._max = 1;
    this._min = 0;
    this.cfg.container = this._el;
  }

  onAdd(map: L.Map): this {
    const size = map.getSize();

    this._map = map;
    this._width = Math.max(size.x, 100);
    this._height = Math.max(size.y, 100);

    this._el.style.width = this._width + 'px';
    this._el.style.height = this._height + 'px';
    this._el.style.position = 'absolute';
    this._el.style.top = '0';
    this._el.style.left = '0';
    this._el.style.zIndex = '200';

    this._origin = this._map.layerPointToLatLng(new L.Point(0, 0));

    map.getPanes().overlayPane?.appendChild(this._el);

    // Defer heatmap creation to ensure container has dimensions
    setTimeout(() => {
      if (!this._heatmap && this._width > 0 && this._height > 0) {
        try {
          this._heatmap = h337.create(this.cfg);
        } catch (e) {
          console.warn('Heatmap creation failed:', e);
        }
      }
      this._draw();
    }, 100);

    map.on('moveend', this._reset, this);
    return this;
  }

  onRemove(map: L.Map): this {
    map.getPanes().overlayPane?.removeChild(this._el);
    map.off('moveend', this._reset, this);
    return this;
  }

  private _draw(): void {
    if (!this._map) return;

    const mapPane = this._map.getPanes().mapPane;
    // @ts-ignore
    const point = mapPane._leaflet_pos;

    if (point) {
      this._el.style.transform = 'translate(' +
        -Math.round(point.x) + 'px,' +
        -Math.round(point.y) + 'px)';
    }

    this._update();
  }

  private _update(): void {
    const generatedData: { max: number; min: number; data: any[] } = { 
      max: this._max, 
      min: this._min, 
      data: [] 
    };

    if (!this._map || !this._heatmap) return;
    
    // Check if container has valid dimensions
    if (this._width <= 0 || this._height <= 0) return;

    const bounds = this._map.getBounds();
    const zoom = this._map.getZoom();
    const scale = Math.pow(2, zoom);

    if (this._data.length === 0) {
      try {
        this._heatmap.setData(generatedData);
      } catch (e) {
        // Ignore canvas errors when dimensions are invalid
      }
      return;
    }

    const latLngPoints: any[] = [];
    const radiusMultiplier = this.cfg.scaleRadius ? scale : 1;
    let localMax = 0;
    let localMin = Infinity;
    const valueField = this.cfg.valueField;

    for (let i = 0; i < this._data.length; i++) {
      const entry = this._data[i];
      const value = entry[valueField];
      const latlng = entry.latlng;

      if (!bounds.contains(latlng)) {
        continue;
      }

      localMax = Math.max(value, localMax);
      localMin = Math.min(value, localMin);

      const point = this._map.latLngToContainerPoint(latlng);
      const latlngPoint: any = { x: Math.round(point.x), y: Math.round(point.y) };
      latlngPoint[valueField] = value;

      let radius: number;
      if (entry.radius) {
        radius = entry.radius * radiusMultiplier;
      } else {
        radius = (this.cfg.radius || 2) * radiusMultiplier;
      }
      latlngPoint.radius = radius;
      latLngPoints.push(latlngPoint);
    }

    if (this.cfg.useLocalExtrema) {
      generatedData.max = localMax;
      generatedData.min = localMin;
    }

    generatedData.data = latLngPoints;

    try {
      this._heatmap.setData(generatedData);
    } catch (e) {
      // Ignore canvas errors when dimensions are invalid
    }
  }

  setData(data: { max?: number; min?: number; data: any[] }): void {
    this._max = data.max || this._max;
    this._min = data.min || this._min;
    const latField = this.cfg.latField || 'lat';
    const lngField = this.cfg.lngField || 'lng';
    const valueField = this.cfg.valueField || 'value';

    const rawData = data.data;
    const d: any[] = [];

    for (let i = 0; i < rawData.length; i++) {
      const entry = rawData[i];
      const latlng = new L.LatLng(entry[latField], entry[lngField]);
      const dataObj: any = { latlng: latlng };
      dataObj[valueField] = entry[valueField];
      if (entry.radius) {
        dataObj.radius = entry.radius;
      }
      d.push(dataObj);
    }
    this._data = d;

    // If heatmap not ready yet, retry after delay
    if (!this._heatmap) {
      setTimeout(() => this._draw(), 150);
    } else {
      this._draw();
    }
  }

  private _reset(): void {
    if (!this._map) return;
    
    this._origin = this._map.layerPointToLatLng(new L.Point(0, 0));

    const size = this._map.getSize();
    const newWidth = Math.max(size.x, 1);
    const newHeight = Math.max(size.y, 1);
    
    if (this._width !== newWidth || this._height !== newHeight) {
      this._width = newWidth;
      this._height = newHeight;

      this._el.style.width = this._width + 'px';
      this._el.style.height = this._height + 'px';

      if (this._heatmap && this._heatmap._renderer) {
        try {
          this._heatmap._renderer.setDimensions(this._width, this._height);
        } catch (e) {
          // Ignore resize errors
        }
      }
    }
    this._draw();
  }
}

export default function LeafletHeatmapInner({ points, minPressure, maxPressure }: LeafletHeatmapInnerProps) {
  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  const heatmapLayerRef = useRef<HeatmapOverlay | null>(null);

  // Initialize map
  useEffect(() => {
    if (!mapRef.current || mapInstanceRef.current) return;

    const map = L.map(mapRef.current, {
      center: [40.7128, -74.006],
      zoom: 13,
      zoomControl: true,
    });

    // Add light tile layer
    L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
      maxZoom: 19,
      attribution: '© CARTO',
    }).addTo(map);

    // Add zoom control
    L.control.zoom({ position: 'bottomright' }).addTo(map);

    // Create heatmap layer with configuration for smooth, realistic appearance
    const cfg = {
      radius: 60,
      maxOpacity: 0.7,
      minOpacity: 0.05,
      blur: 0.9,
      scaleRadius: false,
      useLocalExtrema: true,
      latField: 'lat',
      lngField: 'lng',
      valueField: 'count',
      gradient: {
        0.0: 'rgba(34, 197, 94, 0.0)',
        0.1: 'rgba(34, 197, 94, 0.3)',
        0.25: 'rgba(132, 204, 22, 0.5)',
        0.4: 'rgba(234, 179, 8, 0.6)',
        0.6: 'rgba(245, 158, 11, 0.7)',
        0.8: 'rgba(239, 68, 68, 0.8)',
        1.0: 'rgba(220, 38, 38, 0.9)'
      }
    };

    heatmapLayerRef.current = new HeatmapOverlay(cfg);
    heatmapLayerRef.current.addTo(map);

    mapInstanceRef.current = map;

    return () => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove();
        mapInstanceRef.current = null;
      }
    };
  }, []);

  // Update heatmap and markers when points change
  useEffect(() => {
    if (!mapInstanceRef.current || points.length === 0) return;

    const map = mapInstanceRef.current;

    // Prepare heatmap data
    // Normalize pressure values: lower pressure = higher risk = higher count
    const heatmapData = {
      max: maxPressure,
      min: minPressure,
      data: points.map(point => ({
        lat: point.lat,
        lng: point.lng,
        // Invert so low pressure shows as hot (red), high pressure as cool (green)
        count: maxPressure - point.intensity + minPressure
      }))
    };

    // Update heatmap layer with slight delay to ensure initialization
    const updateHeatmap = () => {
      if (heatmapLayerRef.current) {
        heatmapLayerRef.current.setData(heatmapData);
      }
    };
    
    // Initial update with delay, then immediate updates
    setTimeout(updateHeatmap, 200);

    // Fit bounds to show all points (markers removed for cleaner heatmap view)
    if (points.length > 0) {
      const bounds = L.latLngBounds(points.map(p => [p.lat, p.lng] as L.LatLngTuple));
      map.fitBounds(bounds, { padding: [50, 50] });
    }
  }, [points, minPressure, maxPressure]);

  return (
    <>
      <div ref={mapRef} className="w-full h-full" style={{ minHeight: '400px', height: '100%' }} />
      <style dangerouslySetInnerHTML={{ __html: `
        .leaflet-zoom-hide {
          pointer-events: none;
          z-index: 200 !important;
        }
        .leaflet-zoom-hide canvas {
          position: absolute !important;
          top: 0 !important;
          left: 0 !important;
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
        .leaflet-popup-content-wrapper {
          border-radius: 12px;
          box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        }
        .leaflet-popup-tip {
          background: white;
        }
      `}} />
    </>
  );
}
