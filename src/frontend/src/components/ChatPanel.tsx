import React, { useState, useRef, useEffect } from 'react';
import { Send, Sparkles, AlertCircle, TrendingUp, CheckCircle, X, Home, Plus } from 'lucide-react';
import { Button } from './ui/button';
import { Textarea } from './ui/textarea';
import { ScrollArea } from './ui/scroll-area';
import { Badge } from './ui/badge';
import { Dialog, DialogTrigger, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogClose } from './ui/dialog';
import { Input } from './ui/input';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  formatted?: boolean;
} 

interface Listing {
  id: string;
  url: string;
  price: string;
  location: string;
  predicted_prices?: any;
  prediction_breakdown?: any;
}

interface ChatPanelProps {
  parameters: {
    financial: {
      income: string;
      savings: string;
      budget: string;
      creditScore: string;
      riskTolerance: number;
    };
    priorities: Array<{
      label: string;
      value: number;
      description: string;
    }>;
    listings: Array<{
      location: string;
      price: string;
      url: string;
    }>;
  };
  listings: Listing[];
  setListings: (listings: Listing[]) => void;
}

const quickPrompts = [
  "Compare these 3 listings for me",
  "Can I afford the $750K home?",
  "Which property is the best investment?",
  "What's the stress test on my budget?"
];

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000';

export function ChatPanel({ parameters, listings, setListings }: ChatPanelProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isThinking, setIsThinking] = useState(false);
  const [listingUrl, setListingUrl] = useState('');
  const [isAddDialogOpen, setIsAddDialogOpen] = useState(false);
  const [isScraping, setIsScraping] = useState(false);
  const [isLoadingHistory, setIsLoadingHistory] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Always show demo conversation on load (skip backend history)
  useEffect(() => {
    setMessages([
      {
        id: 'welcome',
        role: 'assistant',
        content: "Welcome to A-list Housings! I'm your AI real estate advisor for the Waterloo Region. I can see you have 3 active listings loaded. Here's what I can help you with:\n\n**Quick Actions:**\n• Compare all your listings side-by-side\n• Run a mortgage stress test on your budget\n• Analyze which property best fits your priorities\n• Estimate 5-year price appreciation\n\nTry asking me something like \"Can I afford the $750K home on University Ave?\" to get started!",
        formatted: true,
      },
      {
        id: 'demo-user-1',
        role: 'user',
        content: 'Compare these 3 listings for me',
        formatted: false,
      },
      {
        id: 'demo-assistant-1',
        role: 'assistant',
        content: "**Listing Comparison — Waterloo Region**\n\n**1. 200 University Ave W, Waterloo — $750,000**\n• 3 bed / 2 bath / 1,600 sqft Single Family\n• Monthly payment (5% down): ~$4,200/mo\n• Down payment needed: $37,500\n• Budget fit: ⚠️ Exceeds your $3,500/mo budget by $700\n\n**2. 10 Victoria St S, Kitchener — $485,000**\n• 2 bed / 1 bath / 900 sqft Condo\n• Monthly payment (5% down): ~$2,700/mo\n• Down payment needed: $24,250\n• Budget fit: ✅ Within your budget with $800/mo cushion\n\n**3. 100 Regina St S, Waterloo — $920,000**\n• 4 bed / 3 bath / 2,400 sqft Single Family\n• Monthly payment (5% down): ~$5,150/mo\n• Down payment needed: $46,000\n• Budget fit: ⚠️ Significantly over budget\n\n**Recommendation:** The Kitchener condo at $485K is the safest financial choice. The University Ave property could work if you increase your down payment to 15%+ to lower monthly costs.",
        formatted: true,
      },
    ]);
    setIsLoadingHistory(false);
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      formatted: false,
    };

    setMessages([...messages, userMessage]);
    const currentInput = input;
    setInput('');
    setIsThinking(true);

    try {
      const response = await fetch(`${API_BASE_URL}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: currentInput,
          parameters: parameters,
        }),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data = await response.json();
      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.content || "I'm sorry, I couldn't generate a response. Please try again.",
        formatted: data.formatted || false,
      };
      setMessages(prev => [...prev, aiMessage]);
    } catch (error) {
      console.error('Error calling chat API:', error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: "I'm sorry, I encountered an error. Please make sure the backend server is running and the Gemini API key is configured.",
        formatted: false,
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsThinking(false);
    }
  };

  const handleQuickPrompt = (prompt: string) => {
    setInput(prompt);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const addListing = async () => {
    if (!listingUrl.trim()) return;
    
    setIsScraping(true);
    const urlToScrape = listingUrl.trim();
    setListingUrl('');
    
    console.log('[CHAT ADD LISTING] Starting scrape for:', urlToScrape);
    console.log('[CHAT ADD LISTING] API URL:', `${API_BASE_URL}/api/scrape-listing`);
    
    try {
      // Call the scraping API
      const response = await fetch(`${API_BASE_URL}/api/scrape-listing`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ url: urlToScrape }),
      });
      
      console.log('[CHAT ADD LISTING] Scrape response status:', response.status);
      
      if (!response.ok) {
        throw new Error(`Scraping failed: ${response.status}`);
      }
      
      const scrapedData = await response.json();
      console.log('[CHAT ADD LISTING] Scraped data:', scrapedData);
      
      // Create listing from scraped data
      const newListing: Listing = {
        id: Date.now().toString(),
        url: scrapedData.url || urlToScrape,
        price: scrapedData.price || 'Price not available',
        location: scrapedData.location || 'Location not found',
        predicted_prices: scrapedData.predicted_prices,
        prediction_breakdown: scrapedData.prediction_breakdown
      };
      
      // Save to database
      console.log('[CHAT ADD LISTING] Saving to database...');
      try {
        const saveResponse = await fetch(`${API_BASE_URL}/api/listings`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ listing: scrapedData }),
        });
        
        console.log('[CHAT ADD LISTING] Save response status:', saveResponse.status);
        
        if (saveResponse.ok) {
          const savedData = await saveResponse.json();
          console.log('[CHAT ADD LISTING] Listing saved to database:', savedData);
          if (savedData.listing?.id) {
            newListing.id = savedData.listing.id.toString();
          }
        } else {
          const errorData = await saveResponse.json();
          console.error('[CHAT ADD LISTING] Failed to save listing to database:', errorData);
        }
      } catch (saveError) {
        console.error('[CHAT ADD LISTING] Error saving listing to database:', saveError);
      }
      
      console.log('[CHAT ADD LISTING] Adding to UI:', newListing);
      setListings([...listings, newListing]);
      setIsAddDialogOpen(false);
    } catch (error) {
      console.error('[CHAT ADD LISTING] Error scraping listing:', error);
      const errorListing: Listing = {
        id: Date.now().toString(),
        url: urlToScrape,
        price: 'Error loading price',
        location: 'Error loading location',
      };
      setListings([...listings, errorListing]);
      setIsAddDialogOpen(false);
    } finally {
      setIsScraping(false);
    }
  };

  const removeListing = (id: string) => {
    setListings(listings.filter(l => l.id !== id));
  };

  const renderMessageContent = (message: Message) => {
    if (!message.formatted) {
      return <p className="whitespace-pre-wrap">{message.content}</p>;
    }

    // Parse formatted content (this is simplified - in production you'd use a proper markdown parser)
    const lines = message.content.split('\n');
    return (
      <div className="space-y-2">
        {lines.map((line, idx) => {
          if (line.startsWith('**') && line.endsWith('**')) {
            return (
              <p key={idx} className="font-semibold text-slate-900">
                {line.replace(/\*\*/g, '')}
              </p>
            );
          }
          if (line.startsWith('•')) {
            return (
              <div key={idx} className="flex gap-2 items-start">
                <span className="text-teal-600 shrink-0">•</span>
                <span className="text-slate-700">{line.substring(1).trim()}</span>
              </div>
            );
          }
          if (line.includes('**') && line.includes(':')) {
            const parts = line.split('**');
            return (
              <p key={idx} className="text-slate-700">
                {parts.map((part, i) => 
                  i % 2 === 1 ? <strong key={i} className="text-slate-900">{part}</strong> : part
                )}
              </p>
            );
          }
          if (line.startsWith('⚠️')) {
            return (
              <div key={idx} className="flex gap-2 items-start text-amber-700 bg-amber-50 px-3 py-2 rounded">
                <AlertCircle className="size-4 shrink-0 mt-0.5" />
                <span className="text-sm">{line.substring(2).trim()}</span>
              </div>
            );
          }
          if (line.includes('Score:')) {
            return (
              <div key={idx} className="flex items-center gap-2 mt-2">
                <Badge className="bg-teal-600 text-white border-0">
                  {line}
                </Badge>
              </div>
            );
          }
          if (line.trim() === '') {
            return <div key={idx} className="h-2" />;
          }
          return (
            <p key={idx} className="text-slate-700">
              {line}
            </p>
          );
        })}
      </div>
    );
  };

  return (
    <div className="size-full flex flex-col bg-white">
      {/* Chat Messages */}
      <div className="border-b border-slate-200 bg-slate-50 px-6 py-3">
          <div className="max-w-3xl mx-auto">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <Home className="size-4 text-teal-600" />
                <span className="text-sm font-medium text-slate-900">Active Listings ({listings.length})</span>
              </div>
              <Dialog>
                <DialogTrigger asChild>
                  <Button 
                    size="sm" 
                    variant="outline" 
                    className="h-7 text-xs"
                  >
                    <Plus className="size-3 mr-1" />
                    Add Property
                  </Button>
                </DialogTrigger>
                <DialogContent className="bg-white text-slate-900">
                  <DialogHeader>
                    <DialogTitle style={{ color: 'black', fontSize: '20px' }}>Add Property Listing</DialogTitle>
                    <DialogDescription style={{ color: '#666', fontSize: '14px' }}>
                      Paste a URL from Zillow, Redfin, or any real estate website to add it to your comparison.
                    </DialogDescription>
                  </DialogHeader>
                  <div className="space-y-4 pt-4">
                    <div className="space-y-2">
                      <Input
                        value={listingUrl}
                        onChange={(e) => setListingUrl(e.target.value)}
                        placeholder="https://zillow.com/homedetails/..."
                        onKeyDown={(e) => e.key === 'Enter' && addListing()}
                        style={{ backgroundColor: 'white', color: 'black', border: '1px solid #ccc', padding: '8px' }}
                      />
                    </div>
                    <div className="flex justify-end gap-2">
                      <DialogClose asChild>
                        <Button variant="outline" style={{ backgroundColor: 'white', color: 'black', border: '1px solid #ccc' }}>
                          Cancel
                        </Button>
                      </DialogClose>
                      <Button onClick={addListing} disabled={!listingUrl.trim()} className="bg-teal-600 hover:bg-teal-700" style={{ backgroundColor: '#0d9488', color: 'white' }}>
                        Add Listing
                      </Button>
                    </div>
                  </div>
                </DialogContent>
              </Dialog>
            </div>
            <div className="flex gap-2 flex-wrap">
              {listings.map((listing) => (
                <div
                  key={listing.id}
                  className="flex items-center gap-2 px-3 py-1.5 rounded-lg border border-slate-200 bg-white text-sm"
                >
                  <div>
                    <span className="font-medium text-slate-900">{listing.price}</span>
                    <span className="text-slate-500 ml-2">{listing.location}</span>
                  </div>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="size-6 -mr-1 hover:bg-rose-50 hover:text-rose-600"
                    onClick={() => removeListing(listing.id)}
                  >
                    <X className="size-3" />
                  </Button>
                </div>
              ))}
            </div>
          </div>
        </div>
      <div className="flex-1 overflow-hidden">
        <ScrollArea className="h-full">
          <div className="p-6">
            <div className="max-w-3xl mx-auto space-y-6">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-[80%] rounded-2xl px-4 py-3 ${
                      message.role === 'user'
                        ? 'bg-teal-600 text-white'
                        : 'bg-white border border-slate-200 text-slate-700 shadow-sm'
                    }`}
                  >
                    {message.role === 'assistant' && (
                      <div className="flex items-center gap-2 mb-2">
                        <div className="size-6 rounded-full bg-gradient-to-br from-teal-500 to-teal-600 flex items-center justify-center">
                          <Sparkles className="size-3.5 text-white" />
                        </div>
                        <span className="text-xs font-medium text-teal-600">AI Advisor</span>
                      </div>
                    )}
                    <div className={message.role === 'user' ? 'text-white' : ''}>
                      {renderMessageContent(message)}
                    </div>
                  </div>
                </div>
              ))}

              {isThinking && (
                <div className="flex justify-start">
                  <div className="max-w-[80%] rounded-2xl px-4 py-3 bg-white border border-slate-200 shadow-sm">
                    <div className="flex items-center gap-3">
                      <div className="size-6 rounded-full bg-gradient-to-br from-teal-500 to-teal-600 flex items-center justify-center">
                        <Sparkles className="size-3.5 text-white animate-pulse" />
                      </div>
                      <div className="flex gap-1">
                        <div className="size-2 rounded-full bg-slate-400 animate-bounce [animation-delay:0ms]" />
                        <div className="size-2 rounded-full bg-slate-400 animate-bounce [animation-delay:150ms]" />
                        <div className="size-2 rounded-full bg-slate-400 animate-bounce [animation-delay:300ms]" />
                      </div>
                    </div>
                  </div>
                </div>
              )}
              
              <div ref={messagesEndRef} />
            </div>
          </div>
        </ScrollArea>
      </div>

      {/* Quick Prompts */}
      <div className="px-6 py-3 border-t border-slate-200 bg-white">
        <div className="max-w-3xl mx-auto flex gap-2 flex-wrap">
          {quickPrompts.map((prompt) => (
            <Button
              key={prompt}
              variant="outline"
              size="sm"
              onClick={() => handleQuickPrompt(prompt)}
              className="text-sm text-slate-600 hover:text-teal-600 hover:border-teal-300"
            >
              {prompt}
            </Button>
          ))}
        </div>
      </div>

      {/* Input Area */}
      <div className="px-6 py-4 border-t border-slate-200 bg-white">
        <div className="max-w-3xl mx-auto flex gap-3">
          <Textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about affordability, trends, or compare listings..."
            className="min-h-[56px] max-h-32 resize-none"
            rows={1}
          />
          <Button
            onClick={sendMessage}
            disabled={!input.trim() || isThinking}
            size="icon"
            className="size-14 shrink-0 bg-teal-600 hover:bg-teal-700"
          >
            <Send className="size-5" />
          </Button>
        </div>
        <p className="text-xs text-slate-500 text-center mt-2 max-w-3xl mx-auto">
          Press Enter to send, Shift+Enter for new line
        </p>
      </div>
    </div>
  );
}