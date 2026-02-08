import { useState, useEffect } from 'react';
import { Settings, MapPin, DollarSign, Shield, GraduationCap, TrendingUp, Wrench } from 'lucide-react';
import { Button } from './components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './components/ui/select';
import { ParametersPanel } from './components/ParametersPanel';
import { ChatPanel } from './components/ChatPanel';
import { MapPanel } from './components/MapPanel';
import { APIProvider } from '@vis.gl/react-google-maps';

const GOOGLE_MAPS_API_KEY = import.meta.env.VITE_GOOGLE_MAPS_API_KEY || '';
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5001';

interface Priority {
  id: string;
  label: string;
  icon: any;
  value: number;
  description: string;
}

interface Listing {
  id: string;
  url: string;
  price: string;
  location: string;
  price_raw?: number;
  address?: string;
  city?: string;
  state?: string;
  zip_code?: string;
  bedrooms?: number;
  bathrooms?: number;
  square_feet?: number;
  property_type?: string;
  year_built?: number;
  lot_size?: number;
  source?: string;
  error?: string;
}

export default function App() {
  // Auth state
  const [user, setUser] = useState<any>(null);
  const [authLoading, setAuthLoading] = useState(true);

  // Check authentication on load
  useEffect(() => {
    const checkAuth = async () => {
      try {
        const res = await fetch('/api/user');
        if (res.ok) {
          const data = await res.json();
          if (data) {
            setUser(data);
          } else {
            window.location.href = '/login';
            return;
          }
        } else {
          window.location.href = '/login';
          return;
        }
      } catch (err) {
        console.error('Auth check failed:', err);
        window.location.href = '/login';
        return;
      } finally {
        setAuthLoading(false);
      }
    };
    checkAuth();
  }, []);

  // Financial profile state
  const [income, setIncome] = useState('120000');
  const [savings, setSavings] = useState('60000');
  const [budget, setBudget] = useState('3500');
  const [creditScore, setCreditScore] = useState('740');
  const [riskTolerance, setRiskTolerance] = useState([50]);

  // Priorities state
  const [priorities, setPriorities] = useState<Priority[]>([
    { id: 'affordability', label: 'Affordability', icon: DollarSign, value: 85, description: 'How important is staying within budget' },
    { id: 'commute', label: 'Commute Time', icon: MapPin, value: 60, description: 'Proximity to work and transport' },
    { id: 'safety', label: 'Neighborhood Safety', icon: Shield, value: 75, description: 'Crime rates and safety ratings' },
    { id: 'schools', label: 'School Quality', icon: GraduationCap, value: 40, description: 'School district ratings' },
    { id: 'investment', label: 'Investment Growth', icon: TrendingUp, value: 55, description: 'Potential for value appreciation' },
    { id: 'renovation', label: 'Renovation Tolerance', icon: Wrench, value: 30, description: 'Willingness to do repairs' },
  ]);

  // Listings state
  const [listings, setListings] = useState<Listing[]>([]);

  // Load saved listings on mount
  useEffect(() => {
    const loadListings = async () => {
      try {
        console.log('[APP] Fetching listings from:', `${API_BASE_URL}/api/listings?user_id=1`);
        const response = await fetch(`${API_BASE_URL}/api/listings?user_id=1`);
        console.log('[APP] Response status:', response.status);
        
        if (response.ok) {
          const data = await response.json();
          console.log('[APP] Received data:', data);
          
          // API returns array directly, not wrapped in object
          const listingsArray = Array.isArray(data) ? data : [];
          
          // Convert backend listings to frontend format
          const loadedListings: Listing[] = listingsArray.map((l: any) => ({
            id: l.id?.toString() || Date.now().toString(),
            url: l.url,
            price: l.price || 'Price not available',
            location: l.location || l.city || 'Unknown location',
            price_raw: l.price_raw,
            address: l.address,
            city: l.city,
            state: l.state,
            zip_code: l.zip_code,
            bedrooms: l.bedrooms,
            bathrooms: l.bathrooms,
            square_feet: l.square_feet,
            property_type: l.property_type,
            year_built: l.year_built,
            lot_size: l.acreage,
            source: l.source,
          }));
          
          setListings(loadedListings);
          console.log('[APP] Loaded listings from backend:', loadedListings.length, loadedListings);
        }
      } catch (error) {
        console.error('[APP] Error loading listings:', error);
      }
    };

    loadListings();
  }, []); // Empty dependency array means this runs once on mount

  // Prepare parameters object for API
  const parameters = {
    financial: {
      income,
      savings,
      budget,
      creditScore,
      riskTolerance: riskTolerance[0],
    },
    priorities: priorities.map(p => ({
      label: p.label,
      value: p.value,
      description: p.description,
    })),
    listings: listings.map(l => ({
      location: l.location,
      price: l.price,
      url: l.url,
      price_raw: l.price_raw,
      address: l.address,
      city: l.city,
      state: l.state,
      zip_code: l.zip_code,
      bedrooms: l.bedrooms,
      bathrooms: l.bathrooms,
      square_feet: l.square_feet,
      property_type: l.property_type,
      year_built: l.year_built,
      lot_size: l.lot_size,
      source: l.source,
    })),
  };

  if (authLoading) {
    return (
      <div style={{ height: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <p>Loading...</p>
      </div>
    );
  }

  return (
    <APIProvider apiKey={GOOGLE_MAPS_API_KEY}>
      <div style={{ height: '100vh' }} className="size-full flex flex-col bg-slate-50">
      {/* Top Navigation */}
      <header className="h-16 border-b border-slate-200 bg-white shadow-sm flex items-center px-6 shrink-0">
        <div className="flex items-center gap-3">
          <div className="size-10 rounded-lg bg-gradient-to-br from-teal-500 to-teal-600 flex items-center justify-center">
            <MapPin className="size-6 text-white" />
          </div>
          <div>
            <h1 className="text-slate-900">placeholder</h1>
            <p className="text-xs text-slate-500">Smart Real Estate Analysis</p>
          </div>
        </div>

        <div className="ml-auto flex items-center gap-4">
          <div className="flex items-center gap-2">
            <span className="text-sm text-slate-600">City:</span>
            <Select defaultValue="seattle">
              <SelectTrigger className="w-40 bg-white">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="seattle">Seattle, WA</SelectItem>
                <SelectItem value="portland">Portland, OR</SelectItem>
                <SelectItem value="san-francisco">San Francisco, CA</SelectItem>
                <SelectItem value="austin">Austin, TX</SelectItem>
                <SelectItem value="denver">Denver, CO</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <Button variant="ghost" size="icon">
            <Settings className="size-5" />
          </Button>
        </div>
      </header>

      {/* 3-Panel Layout */}   
      <div className="flex-1 flex overflow-hidden min-h-0">
        {/* Left Panel - Parameters */}
        <div className="w-80 shrink-0 border-r border-slate-200 bg-white overflow-y-auto min-h-0">
          <ParametersPanel
            income={income}
            setIncome={setIncome}
            savings={savings}
            setSavings={setSavings}
            budget={budget}
            setBudget={setBudget}
            creditScore={creditScore}
            setCreditScore={setCreditScore}
            riskTolerance={riskTolerance}
            setRiskTolerance={setRiskTolerance}
            priorities={priorities}
            setPriorities={setPriorities}
            listings={listings}
            setListings={setListings}
          />
        </div>

        {/* Center Panel - Chat */}
        <div className="flex-1 min-w-0 bg-slate-50 overflow-y-auto min-h-0">
          <ChatPanel parameters={parameters} listings={listings} setListings={setListings} />
        </div>

        {/* Right Panel - Map */}
        <div className="w-[500px] shrink-0 border-l border-slate-200 bg-white overflow-y-auto min-h-0">
          <MapPanel />
        </div>
      </div>
    </div>
    </APIProvider>
  );
}
