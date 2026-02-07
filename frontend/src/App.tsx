import { Settings, MapPin } from 'lucide-react';
import { Button } from './components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './components/ui/select';
import { ParametersPanel } from './components/ParametersPanel';
import { ChatPanel } from './components/ChatPanel';
import { MapPanel } from './components/MapPanel';
import { APIProvider } from '@vis.gl/react-google-maps';

const GOOGLE_MAPS_API_KEY = import.meta.env.VITE_GOOGLE_MAPS_API_KEY || '';

export default function App() {
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
            <h1 className="text-slate-900">HomeAdvisor AI</h1>
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
          <ParametersPanel />
        </div>

        {/* Center Panel - Chat */}
        <div className="flex-1 min-w-0 bg-slate-50 overflow-y-auto min-h-0">
          <ChatPanel />
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
