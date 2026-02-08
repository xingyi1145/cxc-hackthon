import { useState } from 'react';
import { DollarSign, TrendingUp, Shield, GraduationCap, Home, Wrench, MapPin, Plus, X, Info } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Slider } from './ui/slider';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from './ui/collapsible';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from './ui/tooltip';
import { ScrollArea } from './ui/scroll-area';

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

interface ParametersPanelProps {
  income: string;
  setIncome: (value: string) => void;
  savings: string;
  setSavings: (value: string) => void;
  budget: string;
  setBudget: (value: string) => void;
  creditScore: string;
  setCreditScore: (value: string) => void;
  riskTolerance: number[];
  setRiskTolerance: (value: number[]) => void;
  priorities: Priority[];
  setPriorities: (value: Priority[]) => void;
  listings: Listing[];
  setListings: (value: Listing[]) => void;
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000';

export function ParametersPanel({
  income,
  setIncome,
  savings,
  setSavings,
  budget,
  setBudget,
  creditScore,
  setCreditScore,
  riskTolerance,
  setRiskTolerance,
  priorities,
  setPriorities,
  listings,
  setListings,
}: ParametersPanelProps) {
  const [listingUrl, setListingUrl] = useState('');
  const [isScraping, setIsScraping] = useState(false);

  const [financialOpen, setFinancialOpen] = useState(true);
  const [prioritiesOpen, setPrioritiesOpen] = useState(true);
  const [listingsOpen, setListingsOpen] = useState(true);

  const updatePriority = (id: string, value: number) => {
    setPriorities(priorities.map(p => p.id === id ? { ...p, value } : p));
  };

  // Map icon names to actual icons
  const iconMap: Record<string, any> = {
    affordability: DollarSign,
    commute: MapPin,
    safety: Shield,
    schools: GraduationCap,
    investment: TrendingUp,
    renovation: Wrench,
  };

  const addListing = async () => {
    if (!listingUrl.trim()) return;
    
    setIsScraping(true);
    const urlToScrape = listingUrl.trim();
    setListingUrl('');
    
    console.log('[ADD LISTING] Starting scrape for:', urlToScrape);
    console.log('[ADD LISTING] API URL:', `${API_BASE_URL}/api/scrape-listing`);
    
    try {
      // Call the scraping API
      const response = await fetch(`${API_BASE_URL}/api/scrape-listing`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ url: urlToScrape }),
      });
      
      console.log('[ADD LISTING] Scrape response status:', response.status);
      
      if (!response.ok) {
        throw new Error(`Scraping failed: ${response.status}`);
      }
      
      const scrapedData = await response.json();
      console.log('[ADD LISTING] Scraped data:', scrapedData);
      
      // Create listing from scraped data
      const newListing: Listing = {
        id: Date.now().toString(),
        url: scrapedData.url || urlToScrape,
        price: scrapedData.price || 'Price not available',
        location: scrapedData.location || 'Location not found',
        price_raw: scrapedData.price_raw,
        address: scrapedData.address,
        city: scrapedData.city,
        state: scrapedData.state,
        zip_code: scrapedData.zip_code,
        bedrooms: scrapedData.bedrooms,
        bathrooms: scrapedData.bathrooms,
        square_feet: scrapedData.square_feet,
        property_type: scrapedData.property_type,
        year_built: scrapedData.year_built,
        lot_size: scrapedData.lot_size,
        source: scrapedData.source,
        error: scrapedData.error,
      };
      
      // Save to database
      console.log('[ADD LISTING] Saving to database...');
      try {
        const saveResponse = await fetch(`${API_BASE_URL}/api/listings`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ listing: scrapedData }),
        });
        
        console.log('[ADD LISTING] Save response status:', saveResponse.status);
        
        if (saveResponse.ok) {
          const savedData = await saveResponse.json();
          console.log('[ADD LISTING] Listing saved to database:', savedData);
          // Update listing with database ID if returned
          if (savedData.listing?.id) {
            newListing.id = savedData.listing.id.toString();
          }
        } else {
          const errorData = await saveResponse.json();
          console.error('[ADD LISTING] Failed to save listing to database:', errorData);
        }
      } catch (saveError) {
        console.error('[ADD LISTING] Error saving listing to database:', saveError);
        // Continue anyway - listing will still appear in UI
      }
      
      console.log('[ADD LISTING] Adding to UI:', newListing);
      setListings([...listings, newListing]);
    } catch (error) {
      console.error('[ADD LISTING] Error scraping listing:', error);
      // Add listing with error info
      const errorListing: Listing = {
        id: Date.now().toString(),
        url: urlToScrape,
        price: 'Error loading price',
        location: 'Error loading location',
        error: error instanceof Error ? error.message : 'Unknown error',
      };
      setListings([...listings, errorListing]);
    } finally {
      setIsScraping(false);
    }
  };

  const removeListing = (id: string) => {
    setListings(listings.filter(l => l.id !== id));
  };

  const getAffordabilityStatus = () => {
    const monthlyIncome = parseFloat(income) / 12;
    const budgetAmount = parseFloat(budget);
    const ratio = budgetAmount / monthlyIncome;
    
    if (ratio < 0.28) return { label: 'Safe', color: 'bg-emerald-500' };
    if (ratio < 0.35) return { label: 'Moderate', color: 'bg-amber-500' };
    return { label: 'Stretch', color: 'bg-rose-500' };
  };

  const affordability = getAffordabilityStatus();

  return (
    <ScrollArea className="h-full w-full">
      <div className="p-4 space-y-4 min-w-0">
        {/* Financial Profile */}
        <Collapsible open={financialOpen} onOpenChange={setFinancialOpen}>
          <Card>
            <CollapsibleTrigger className="w-full">
              <CardHeader className="cursor-pointer hover:bg-slate-50/50 transition-colors">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-base">Financial Profile</CardTitle>
                  <Badge className={`${affordability.color} text-white border-0`}>
                    {affordability.label}
                  </Badge>
                </div>
                <CardDescription>Your income and budget details</CardDescription>
              </CardHeader>
            </CollapsibleTrigger>
            <CollapsibleContent>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="income">Annual Income</Label>
                  <div className="relative">
                    <span className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500">$</span>
                    <Input
                      id="income"
                      type="number"
                      value={income}
                      onChange={(e) => setIncome(e.target.value)}
                      className="pl-7"
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="savings">Savings / Down Payment</Label>
                  <div className="relative">
                    <span className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500">$</span>
                    <Input
                      id="savings"
                      type="number"
                      value={savings}
                      onChange={(e) => setSavings(e.target.value)}
                      className="pl-7"
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="budget">Monthly Housing Budget</Label>
                  <div className="relative">
                    <span className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500">$</span>
                    <Input
                      id="budget"
                      type="number"
                      value={budget}
                      onChange={(e) => setBudget(e.target.value)}
                      className="pl-7"
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="credit">Credit Score (Optional)</Label>
                  <Input
                    id="credit"
                    type="number"
                    value={creditScore}
                    onChange={(e) => setCreditScore(e.target.value)}
                    placeholder="e.g., 740"
                  />
                </div>

                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <Label>Risk Tolerance</Label>
                    <span className="text-sm text-slate-600">
                      {riskTolerance[0] < 35 ? 'Conservative' : riskTolerance[0] > 65 ? 'Aggressive' : 'Balanced'}
                    </span>
                  </div>
                  <Slider
                    value={riskTolerance}
                    onValueChange={setRiskTolerance}
                    max={100}
                    step={1}
                    className="py-2"
                  />
                  <div className="flex justify-between text-xs text-slate-500">
                    <span>Conservative</span>
                    <span>Balanced</span>
                    <span>Aggressive</span>
                  </div>  
                </div>
              </CardContent>
            </CollapsibleContent>
          </Card>
        </Collapsible>

        {/* Priority Toggles */}
        <Collapsible open={prioritiesOpen} onOpenChange={setPrioritiesOpen}>
          <Card>
            <CollapsibleTrigger className="w-full">
              <CardHeader className="cursor-pointer hover:bg-slate-50/50 transition-colors">
                <CardTitle className="text-base">Priorities</CardTitle>
                <CardDescription>Weight factors that matter most to you</CardDescription>
              </CardHeader>
            </CollapsibleTrigger>
            <CollapsibleContent>
              <CardContent className="space-y-5">
                {priorities.map((priority) => {
                  const Icon = iconMap[priority.id] || DollarSign;
                  return (
                    <div key={priority.id} className="space-y-2">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <div className="size-8 rounded-md bg-teal-50 flex items-center justify-center">
                            <Icon className="size-4 text-teal-600" />
                          </div>
                          <Label className="cursor-pointer">{priority.label}</Label>
                          <TooltipProvider>
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <Info className="size-4 text-slate-400" />
                              </TooltipTrigger>
                              <TooltipContent>
                                <p className="text-xs">{priority.description}</p>
                              </TooltipContent>
                            </Tooltip>
                          </TooltipProvider>
                        </div>
                        <span className="text-sm font-medium text-teal-600">{priority.value}%</span>
                      </div>
                      <Slider
                        value={[priority.value]}
                        onValueChange={([value]) => updatePriority(priority.id, value)}
                        max={100}
                        step={5}
                        className="py-1"
                      />
                    </div>
                  );
                })}
              </CardContent>
            </CollapsibleContent>
          </Card>
        </Collapsible>

      </div>
    </ScrollArea>
  );
}
