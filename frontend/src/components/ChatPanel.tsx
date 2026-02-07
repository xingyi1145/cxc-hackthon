import { useState, useRef, useEffect } from 'react';
import { Send, Sparkles, AlertCircle, TrendingUp, CheckCircle } from 'lucide-react';
import { Button } from './ui/button';
import { Textarea } from './ui/textarea';
import { ScrollArea } from './ui/scroll-area';
import { Badge } from './ui/badge';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  formatted?: boolean;
}

const quickPrompts = [
  "Is this affordable?",
  "Best option for my priorities",
  "Market outlook next 6 months"
];

export function ChatPanel() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      role: 'assistant',
      content: "Hi! I'm your AI real estate advisor. I'll help you analyze properties based on your financial profile and priorities. Add some listings to get started, or ask me anything about the housing market!",
      formatted: false,
    },
    {
      id: '2',
      role: 'user',
      content: "What do you think about the Capitol Hill listing?",
      formatted: false,
    },
    {
      id: '3',
      role: 'assistant',
      content: `Let me analyze the **Capitol Hill property at $425,000** based on your profile:

**Financial Fit:**
• Monthly payment: ~$2,850 (with 20% down)
• This is **within your $3,500 budget** ✓
• Down payment needed: $85,000 (you have $60,000)

**Key Considerations:**
• **Affordability (85% priority)**: Strong match - leaves room in your budget
• **Neighborhood Safety (75% priority)**: Capitol Hill scores well on safety metrics
• **Commute Time (60% priority)**: Excellent transit access to downtown

**Potential Concerns:**
⚠️ You're $25,000 short on the down payment - consider PMI or bridging options
⚠️ Capitol Hill market is competitive - properties move quickly

**Overall Score: 78/100** - This is a solid option that aligns well with your top priorities!`,
      formatted: true,
    },
  ]);
  const [input, setInput] = useState('');
  const [isThinking, setIsThinking] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const sendMessage = () => {
    if (!input.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      formatted: false,
    };

    setMessages([...messages, userMessage]);
    setInput('');
    setIsThinking(true);

    // Simulate AI response
    setTimeout(() => {
      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: "I'd be happy to help with that! Based on your current financial profile and priorities, here are my thoughts...",
        formatted: false,
      };
      setMessages(prev => [...prev, aiMessage]);
      setIsThinking(false);
    }, 1500);
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