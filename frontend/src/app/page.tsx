"use client";

import React, { useState } from 'react';
import ChatInterface from '@/components/ChatInterface';
import FileUploader from '@/components/FileUploader';
import { Video, Sparkles } from 'lucide-react';

export default function Home() {
  // We don't need videoRef or videoSrc state for the player anymore 
  // since the video is hidden.
  
  // Logic to handle seek can be kept empty or removed since there is no player.
  const handleSeek = (timestamp: number) => {
    console.log(`Seek requested to: ${timestamp}s (Video player is hidden)`);
  };

  const handleUploadComplete = (file: File) => {
    console.log("File uploaded:", file.name);
  };

  return (
    <main className="min-h-screen bg-slate-950 text-slate-200 p-4 md:p-8 flex flex-col gap-6">
      
      {/* Header */}
      <header className="flex items-center justify-between border-b border-slate-800 pb-4">
        <div className="flex items-center gap-3">
          <div className="p-2.5 bg-blue-600 rounded-xl shadow-lg shadow-blue-900/20">
            <Video className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-cyan-300 bg-clip-text text-transparent">
              VideoRAG
            </h1>
            <p className="text-xs text-slate-500 font-medium tracking-wide">
              AI VIDEO KNOWLEDGE ENGINE
            </p>
          </div>
        </div>
      </header>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 flex-1 min-h-0">
        
        {/* Left Column: Uploader (Takes 4 cols on large screens) */}
        <div className="lg:col-span-4 flex flex-col gap-6">
          <div className="bg-slate-900/50 p-6 rounded-2xl border border-slate-800 shadow-xl backdrop-blur-sm">
             <h2 className="text-sm font-semibold text-slate-400 mb-4 uppercase tracking-wider flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-yellow-500" />
                Ingestion Pipeline
             </h2>
             <FileUploader onUploadComplete={handleUploadComplete} />
             
             <div className="mt-6 text-xs text-slate-500 space-y-2">
                <p>Upload a video to start the indexing process.</p>
                <p>The system will extract:</p>
                <ul className="list-disc pl-4 space-y-1 text-slate-400">
                    <li>Visual Embeddings (CLIP)</li>
                    <li>Audio Transcripts (Whisper)</li>
                    <li>Knowledge Graph (Entities)</li>
                </ul>
             </div>
          </div>
        </div>

        {/* Right Column: Chat Interface (Takes 8 cols - Main Focus) */}
        <div className="lg:col-span-8 h-[600px] lg:h-auto min-h-[500px]">
          <ChatInterface onSeek={handleSeek} />
        </div>
      </div>
    </main>
  );
}