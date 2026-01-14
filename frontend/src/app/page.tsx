"use client";

import React, { useState, useRef } from 'react';
import ChatInterface from '@/components/ChatInterface';
import FileUploader from '@/components/FileUploader';
import { Video } from 'lucide-react';

export default function Home() {
  const [videoSrc, setVideoSrc] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  const handleUploadComplete = (file: File) => {
    // Create a local URL for the video player instantly
    const url = URL.createObjectURL(file);
    setVideoSrc(url);
  };

  const handleSeek = (timestamp: number) => {
    if (videoRef.current) {
      videoRef.current.currentTime = timestamp;
      videoRef.current.play();
    }
  };

  return (
    <main className="min-h-screen p-4 md:p-8 flex flex-col gap-6">
      <header className="flex items-center gap-3 mb-2">
        <div className="p-2 bg-blue-600 rounded-lg">
          <Video className="w-6 h-6 text-white" />
        </div>
        <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-cyan-300 bg-clip-text text-transparent">
          VideoRAG
        </h1>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 h-[calc(100vh-140px)]">
        {/* Left Column: Video Player & Uploader */}
        <div className="flex flex-col gap-6 h-full">
          {/* Video Player Container */}
          <div className="flex-1 bg-black rounded-xl overflow-hidden shadow-2xl border border-slate-800 relative group">
            {videoSrc ? (
              <video
                ref={videoRef}
                src={videoSrc}
                className="w-full h-full object-contain"
                controls
              />
            ) : (
              <div className="w-full h-full flex items-center justify-center bg-slate-900 text-slate-500">
                <p>No video loaded</p>
              </div>
            )}
          </div>

          {/* Uploader Section - Fixed height at bottom of left col */}
          <div className="h-auto">
            <FileUploader onUploadComplete={handleUploadComplete} />
          </div>
        </div>

        {/* Right Column: Chat */}
        <div className="h-full">
          <ChatInterface onSeek={handleSeek} />
        </div>
      </div>
    </main>
  );
}