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
}

export function ParametersPanel() {
  const [income, setIncome] = useState('120000');
  const [savings, setSavings] = useState('60000');
  const [budget, setBudget] = useState('3500');
  const [creditScore, setCreditScore] = useState('740');
  const [riskTolerance, setRiskTolerance] = useState([50]);
  const [listingUrl, setListingUrl] = useState('');
  
  const [priorities, setPriorities] = useState<Priority[]>([
    { id: 'affordability', label: 'Affordability', icon: DollarSign, value: 85, description: 'How important is staying within budget' },
    { id: 'commute', label: 'Commute Time', icon: MapPin, value: 60, description: 'Proximity to work and transport' },
    { id: 'safety', label: 'Neighborhood Safety', icon: Shield, value: 75, description: 'Crime rates and safety ratings' },
    { id: 'schools', label: 'School Quality', icon: GraduationCap, value: 40, description: 'School district ratings' },
    { id: 'investment', label: 'Investment Growth', icon: TrendingUp, value: 55, description: 'Potential for value appreciation' },
    { id: 'renovation', label: 'Renovation Tolerance', icon: Wrench, value: 30, description: 'Willingness to do repairs' },
  ]);

  const [listings, setListings] = useState<Listing[]>([
    { id: '1', url: 'https://zillow.com/...', price: '$425,000', location: 'Capitol Hill' },
    { id: '2', url: 'https://redfin.com/...', price: '$510,000', location: 'Fremont' },
  ]);

  const [financialOpen, setFinancialOpen] = useState(true);
  const [prioritiesOpen, setPrioritiesOpen] = useState(true);
  const [listingsOpen, setListingsOpen] = useState(true);

  const updatePriority = (id: string, value: number) => {
    setPriorities(priorities.map(p => p.id === id ? { ...p, value } : p));
  };

  const addListing = () => {
    if (listingUrl.trim()) {
      const newListing: Listing = {
        id: Date.now().toString(),
        url: listingUrl,
        price: '$' + Math.floor(Math.random() * 500000 + 300000).toLocaleString(),
        location: 'Location TBD',
      };
      setListings([...listings, newListing]);
      setListingUrl('');
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
    <ScrollArea className="h-full">
      <div className="p-4 space-y-4">
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
                  const Icon = priority.icon;
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

        {/* Housing Listings */}
        <Collapsible open={listingsOpen} onOpenChange={setListingsOpen}>
          <Card>
            <CollapsibleTrigger className="w-full">
              <CardHeader className="cursor-pointer hover:bg-slate-50/50 transition-colors">
                <CardTitle className="text-base">Housing Listings</CardTitle>
                <CardDescription>{listings.length} properties added</CardDescription>
              </CardHeader>
            </CollapsibleTrigger>
            <CollapsibleContent>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="listing-url">Property URL</Label>
                  <div className="flex gap-2">
                    <Input
                      id="listing-url"
                      value={listingUrl}
                      onChange={(e) => setListingUrl(e.target.value)}
                      placeholder="https://zillow.com/..."
                      onKeyDown={(e) => e.key === 'Enter' && addListing()}
                    />
                    <Button onClick={addListing} size="icon" className="shrink-0">
                      <Plus className="size-4" />
                    </Button>
                  </div>
                </div>

                <div className="space-y-2">
                  {listings.map((listing) => (
                    <div
                      key={listing.id}
                      className="flex items-start gap-3 p-3 rounded-lg border border-slate-200 bg-white hover:border-teal-300 transition-colors"
                    >
                      <div className="size-12 rounded bg-slate-100 shrink-0 flex items-center justify-center">
                        <Home className="size-6 text-slate-400" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="font-medium text-slate-900">{listing.price}</p>
                        <p className="text-sm text-slate-500 truncate">{listing.location}</p>
                        <p className="text-xs text-slate-400 truncate">{listing.url}</p>
                      </div>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="size-8 shrink-0"
                        onClick={() => removeListing(listing.id)}
                      >
                        <X className="size-4" />
                      </Button>
                    </div>
                  ))}
                </div>
              </CardContent>
            </CollapsibleContent>
          </Card>
        </Collapsible>
      </div>
    </ScrollArea>
  );
}
