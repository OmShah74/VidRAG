"use client";

import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, PlayCircle } from 'lucide-react';
import { sendChatQuery } from '@/lib/api';

interface Source {
    start: number;
    end: number;
    text: string;
}

interface Message {
    role: 'user' | 'bot';
    content: string;
    sources?: Source[];
}

interface ChatInterfaceProps {
    onSeek: (timestamp: number) => void;
}

export default function ChatInterface({ onSeek }: ChatInterfaceProps) {
    const [messages, setMessages] = useState<Message[]>([
        { role: 'bot', content: 'Hello! Upload a video and ask me anything about it.' }
    ]);
    const [input, setInput] = useState("");
    const [isLoading, setIsLoading] = useState(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleSend = async () => {
        if (!input.trim() || isLoading) return;

        const userMsg: Message = { role: 'user', content: input };
        setMessages(prev => [...prev, userMsg]);
        setInput("");
        setIsLoading(true);

        try {
            const response = await sendChatQuery(userMsg.content);
            const botMsg: Message = {
                role: 'bot',
                content: response.data.answer,
                sources: response.data.sources
            };
            setMessages(prev => [...prev, botMsg]);
        } catch (error) {
            setMessages(prev => [...prev, { role: 'bot', content: "Sorry, I encountered an error processing your request." }]);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="flex flex-col h-full bg-slate-900 rounded-xl border border-slate-700 overflow-hidden shadow-xl">
            {/* Header */}
            <div className="p-4 border-b border-slate-700 bg-slate-800">
                <h2 className="font-semibold text-slate-100 flex items-center gap-2">
                    <Bot className="w-5 h-5 text-blue-400" />
                    AI Video Assistant
                </h2>
            </div>

            {/* Messages Area */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4 scrollbar-default">
                {messages.map((msg, idx) => (
                    <div key={idx} className={`flex gap-3 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
                        <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${msg.role === 'user' ? 'bg-blue-600' : 'bg-slate-700'}`}>
                            {msg.role === 'user' ? <User className="w-5 h-5" /> : <Bot className="w-5 h-5" />}
                        </div>

                        <div className={`max-w-[80%] rounded-2xl p-3 text-sm leading-relaxed ${msg.role === 'user'
                                ? 'bg-blue-600 text-white rounded-tr-none'
                                : 'bg-slate-800 text-slate-200 rounded-tl-none border border-slate-700'
                            }`}>
                            <p className="whitespace-pre-wrap">{msg.content}</p>

                            {/* Sources / Citations */}
                            {msg.sources && msg.sources.length > 0 && (
                                <div className="mt-3 pt-3 border-t border-slate-700">
                                    <p className="text-xs text-slate-400 font-semibold mb-2">Relevant Segments:</p>
                                    <div className="flex flex-wrap gap-2">
                                        {msg.sources.map((src, i) => (
                                            <button
                                                key={i}
                                                onClick={() => onSeek(src.start)}
                                                className="flex items-center gap-1.5 text-xs bg-slate-700 hover:bg-slate-600 text-blue-300 px-2 py-1 rounded transition-colors border border-slate-600"
                                            >
                                                <PlayCircle className="w-3 h-3" />
                                                {Math.floor(src.start)}s - {Math.floor(src.end)}s
                                            </button>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                ))}
                {isLoading && (
                    <div className="flex gap-3">
                        <div className="w-8 h-8 rounded-full bg-slate-700 flex items-center justify-center">
                            <Bot className="w-5 h-5" />
                        </div>
                        <div className="bg-slate-800 rounded-2xl p-4 border border-slate-700">
                            <div className="flex space-x-2">
                                <div className="w-2 h-2 bg-slate-500 rounded-full animate-bounce"></div>
                                <div className="w-2 h-2 bg-slate-500 rounded-full animate-bounce delay-75"></div>
                                <div className="w-2 h-2 bg-slate-500 rounded-full animate-bounce delay-150"></div>
                            </div>
                        </div>
                    </div>
                )}
                <div ref={messagesEndRef} />
            </div>

            {/* Input Area */}
            <div className="p-4 bg-slate-800 border-t border-slate-700">
                <div className="flex gap-2">
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                        placeholder="Ask about the video context..."
                        className="flex-1 bg-slate-900 border border-slate-600 text-slate-200 rounded-lg px-4 py-3 focus:outline-none focus:border-blue-500 transition-colors"
                    />
                    <button
                        onClick={handleSend}
                        disabled={isLoading || !input.trim()}
                        className="bg-blue-600 hover:bg-blue-700 text-white rounded-lg px-4 py-2 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        <Send className="w-5 h-5" />
                    </button>
                </div>
            </div>
        </div>
    );
}