import { useState } from 'react';
import { MapPin, Layers, Activity, Maximize2, TrendingUp, DollarSign } from 'lucide-react';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Card } from './ui/card';
import { Slider } from './ui/slider';
import { Label } from './ui/label';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from './ui/tooltip';
import { Switch } from './ui/switch';
import { Map, Marker } from '@vis.gl/react-google-maps';

interface MapListing {
  id: string;
  price: number;
  location: string;
  coordinates: { x: number; y: number };
  monthlyPayment: number;
  score: number;
  affordability: 'safe' | 'stretch' | 'high-risk';
}

const mockListings: MapListing[] = [
  {
    id: '1',
    price: 425000,
    location: 'Capitol Hill',
    coordinates: { x: 45, y: 40 },
    monthlyPayment: 2850,
    score: 78,
    affordability: 'safe',
  },
  {
    id: '2',
    price: 510000,
    location: 'Fremont',
    coordinates: { x: 38, y: 30 },
    monthlyPayment: 3420,
    score: 65,
    affordability: 'stretch',
  },
  {
    id: '3',
    price: 380000,
    location: 'Georgetown',
    coordinates: { x: 35, y: 65 },
    monthlyPayment: 2540,
    score: 82,
    affordability: 'safe',
  },
  {
    id: '4',
    price: 625000,
    location: 'Queen Anne',
    coordinates: { x: 42, y: 35 },
    monthlyPayment: 4180,
    score: 58,
    affordability: 'high-risk',
  },
  {
    id: '5',
    price: 455000,
    location: 'Ballard',
    coordinates: { x: 30, y: 25 },
    monthlyPayment: 3050,
    score: 72,
    affordability: 'safe',
  },
  {
    id: '6',
    price: 495000,
    location: 'Wallingford',
    coordinates: { x: 48, y: 32 },
    monthlyPayment: 3320,
    score: 69,
    affordability: 'stretch',
  },
];

export function MapPanel() {
  const [selectedListing, setSelectedListing] = useState<string | null>(null);
  const [hoveredListing, setHoveredListing] = useState<string | null>(null);
  const [overlays, setOverlays] = useState({
    priceHeatmap: false,
    demand: false,
    rentGrowth: false,
  });
  const [priceRange, setPriceRange] = useState([300000, 700000]);
  const [minScore, setMinScore] = useState([60]);

  const toggleOverlay = (key: keyof typeof overlays) => {
    setOverlays({ ...overlays, [key]: !overlays[key] });
  };

  const getAffordabilityColor = (affordability: string) => {
    switch (affordability) {
      case 'safe':
        return 'bg-emerald-500 border-emerald-600';
      case 'stretch':
        return 'bg-amber-500 border-amber-600';
      case 'high-risk':
        return 'bg-rose-500 border-rose-600';
      default:
        return 'bg-slate-500 border-slate-600';
    }
  };

  const filteredListings = mockListings.filter(
    (listing) =>
      listing.price >= priceRange[0] &&
      listing.price <= priceRange[1] &&
      listing.score >= minScore[0]
  );

  const hoveredData = hoveredListing ? mockListings.find(l => l.id === hoveredListing) : null;

  return (
    <div className="size-full flex flex-col bg-slate-100">
      {/* Map Controls Toolbar */}
      <div className="p-3 bg-white border-b border-slate-200 space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Layers className="size-4 text-slate-600" />
            <span className="text-sm font-medium text-slate-700">Map Overlays</span>
          </div>
          <div className="flex gap-2">
            <Button variant="outline" size="icon" className="size-8">
              <Maximize2 className="size-4" />
            </Button>
          </div>
        </div>

        <div className="flex flex-wrap gap-2">
          <Button
            variant={overlays.priceHeatmap ? 'default' : 'outline'}
            size="sm"
            onClick={() => toggleOverlay('priceHeatmap')}
            className={overlays.priceHeatmap ? 'bg-teal-600 hover:bg-teal-700' : ''}
          >
            <DollarSign className="size-3.5 mr-1.5" />
            Price Trends
          </Button>
          <Button
            variant={overlays.demand ? 'default' : 'outline'}
            size="sm"
            onClick={() => toggleOverlay('demand')}
            className={overlays.demand ? 'bg-teal-600 hover:bg-teal-700' : ''}
          >
            <TrendingUp className="size-3.5 mr-1.5" />
            Demand
          </Button>
          <Button
            variant={overlays.rentGrowth ? 'default' : 'outline'}
            size="sm"
            onClick={() => toggleOverlay('rentGrowth')}
            className={overlays.rentGrowth ? 'bg-teal-600 hover:bg-teal-700' : ''}
          >
            <Activity className="size-3.5 mr-1.5" />
            Rent Growth
          </Button>
        </div>
      </div>

      {/* Map Display */}
      <div className="flex-1 relative overflow-hidden">
        <Map
          defaultCenter={{ lat: 43.4643, lng: -80.5204 }}
          defaultZoom={13}
          gestureHandling="greedy"
          disableDefaultUI={false}
          className="w-full h-full"
        >
          {/* Listing Markers */}
          {filteredListings.map((listing) => {
            // Convert percentage coordinates to actual lat/lng relative to Waterloo center
            // This is a simple approximation - adjust as needed for your data
            const latOffset = (listing.coordinates.y - 50) * 0.01; // Rough conversion
            const lngOffset = (listing.coordinates.x - 50) * 0.01;
            const markerLat = 43.4643 + latOffset;
            const markerLng = -80.5204 + lngOffset;
            
            const isSelected = selectedListing === listing.id;
            
            return (
              <Marker
                key={listing.id}
                position={{ lat: markerLat, lng: markerLng }}
                onClick={() => setSelectedListing(listing.id)}
                title={`${listing.location} - $${listing.price.toLocaleString()}`}
              />
            );
          })}
        </Map>

        {/* Legend */}
        <Card className="absolute top-4 left-4 p-3 shadow-lg">
          <p className="text-xs font-medium text-slate-700 mb-2">Affordability</p>
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <div className="size-3 rounded-full bg-emerald-500" />
              <span className="text-xs text-slate-600">Safe</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="size-3 rounded-full bg-amber-500" />
              <span className="text-xs text-slate-600">Stretch</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="size-3 rounded-full bg-rose-500" />
              <span className="text-xs text-slate-600">High Risk</span>
            </div>
          </div>
        </Card>

        {/* Listing Count */}
        <Badge className="absolute top-4 right-4 bg-white text-slate-700 border border-slate-200">
          {filteredListings.length} listings
        </Badge>
      </div>

      {/* Filter Controls */}
      <div className="p-4 bg-white border-t border-slate-200 space-y-4">
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label className="text-xs">Price Range</Label>
            <span className="text-xs text-slate-600">
              ${(priceRange[0] / 1000).toFixed(0)}K - ${(priceRange[1] / 1000).toFixed(0)}K
            </span>
          </div>
          <Slider
            value={priceRange}
            onValueChange={setPriceRange}
            min={200000}
            max={800000}
            step={10000}
            className="py-1"
          />
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label className="text-xs">Minimum Score</Label>
            <span className="text-xs text-slate-600">{minScore[0]}/100</span>
          </div>
          <Slider
            value={minScore}
            onValueChange={setMinScore}
            min={0}
            max={100}
            step={5}
            className="py-1"
          />
        </div>
      </div>
    </div>
  );
}