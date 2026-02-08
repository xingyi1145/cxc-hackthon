import { useState } from 'react';
import { MapPin, Layers, Activity, Maximize2, TrendingUp, DollarSign } from 'lucide-react';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Card } from './ui/card';
import { Slider } from './ui/slider';
import { Label } from './ui/label';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from './ui/tooltip';
import { Switch } from './ui/switch';
import { Map, AdvancedMarker } from '@vis.gl/react-google-maps';

import listingsData from '../data/listings.json'

// Interface for the raw data from listings.json
interface RawListingData {
    id: number;
    city: string;
    address: string;
    longitude: number;
    latitude: number;
    price: string;
    bedrooms: string;
    bathoom: string;
}

// Interface for the processed listings used by the map component
interface DisplayListing {
    id: string;
    price: number;
    location: string; // e.g., Address
    coordinates: { lat: number; lng: number };
    monthlyPayment: number;
    score: number;
    affordability: 'safe' | 'stretch' | 'high-risk';
}

// Process the raw listings data into the DisplayListing format
const processedListings: DisplayListing[] = (listingsData as unknown as RawListingData[])
  .filter(
    (item) =>
      typeof item.longitude === 'number' && typeof item.latitude === 'number' && item.price !== undefined
  )
  .map((item, index) => {
    // Clean and parse the price string
    const numericPrice = parseFloat(item.price.replace(/[^0-9.-]+/g, ""));

    // Dummy logic for score and affordability, as these are not in listings.json
    let score: number;
    let affordability: 'safe' | 'stretch' | 'high-risk';

    if (numericPrice < 400000) {
      score = Math.floor(Math.random() * (90 - 70 + 1)) + 70; // 70-90
      affordability = 'safe';
    } else if (numericPrice >= 400000 && numericPrice < 700000) {
      score = Math.floor(Math.random() * (70 - 50 + 1)) + 50; // 50-70
      affordability = 'stretch';
    } else {
      score = Math.floor(Math.random() * (50 - 30 + 1)) + 30; // 30-50
      affordability = 'high-risk';
    }

    return {
      id: `listing-${index}`, // Generate a unique ID
      price: numericPrice,
      location: item.address,
      coordinates: { lat: item.latitude, lng: item.longitude },
      monthlyPayment: Math.round(numericPrice * 0.006), // A simple dummy calculation
      score: score,
      affordability: affordability,
    };
  });

export function MapPanel() {
  const [selectedListing, setSelectedListing] = useState<string | null>(null);
  const [hoveredListing, setHoveredListing] = useState<string | null>(null);
  const [overlays, setOverlays] = useState({
    priceHeatmap: false,
    demand: false,
    rentGrowth: false,
  });
  const [priceRange, setPriceRange] = useState([0, 1000000000]); // Initial broad range
  const [minScore, setMinScore] = useState([0]); // minScore is a single value, not a range

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

  // Apply filtering to the processed listings
  const filteredListings = processedListings;

  const hoveredData = hoveredListing ? processedListings.find(l => l.id === hoveredListing) : null;

  return (
    <div className="size-full flex flex-col bg-slate-100">
      {/* Map Controls Toolbar */}
      {/*
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
      */}

      {/* Map Display */}
      <div className="flex-1 relative overflow-hidden">
        <Map
          defaultCenter={{ lat: 43.4643, lng: -80.5204 }}
          defaultZoom={13}
          mapId="DEMO_MAP_ID"
          gestureHandling="greedy"
          disableDefaultUI={false}
          className="w-full h-full"
        >
          {/* Listing Markers */}
          {filteredListings.map((listing) => {
            // Use actual lat/lng from processed data
            const markerLat = listing.coordinates.lat;
            const markerLng = listing.coordinates.lng;
            
            const isSelected = selectedListing === listing.id;
            
            return (
              <AdvancedMarker
                key={listing.id}
                position={{ lat: markerLat, lng: markerLng }}
                onClick={() => setSelectedListing(listing.id)}
                title={`${listing.location} - $${listing.price.toLocaleString()}`}
              />
            );
          })}
        </Map>

        {/* Legend */}
        {/*
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
        */}

        {/* Listing Count */}
        <Badge className="absolute top-4 right-4 bg-white text-slate-700 border border-slate-200">
          {filteredListings.length} listings
        </Badge>
      </div>

      {/* Filter Controls */}
      <div className="p-4 bg-white border-t border-slate-200 space-y-4">
        {/*
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
        */}
      </div>
    </div>
  );
}