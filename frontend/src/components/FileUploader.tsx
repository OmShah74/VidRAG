"use client";

import React, { useState } from 'react';
import { Upload, Loader2, CheckCircle } from 'lucide-react';
import { uploadVideo } from '@/lib/api';

interface FileUploaderProps {
    onUploadComplete: (file: File) => void;
}

export default function FileUploader({ onUploadComplete }: FileUploaderProps) {
    const [isUploading, setIsUploading] = useState(false);
    const [uploadStatus, setUploadStatus] = useState<'idle' | 'success' | 'error'>('idle');

    const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            const file = e.target.files[0];
            setIsUploading(true);
            setUploadStatus('idle');

            try {
                // Send to Backend
                await uploadVideo(file);

                // Pass file back to parent to show in player
                onUploadComplete(file);
                setUploadStatus('success');
            } catch (error) {
                console.error("Upload failed", error);
                setUploadStatus('error');
            } finally {
                setIsUploading(false);
            }
        }
    };

    return (
        <div className="w-full p-6 border-2 border-dashed border-slate-600 rounded-xl bg-slate-800/50 hover:bg-slate-800 transition-all text-center">
            <input
                type="file"
                id="video-upload"
                accept="video/*"
                className="hidden"
                onChange={handleFileChange}
                disabled={isUploading}
            />

            <label htmlFor="video-upload" className="cursor-pointer flex flex-col items-center justify-center gap-3">
                {isUploading ? (
                    <>
                        <Loader2 className="w-10 h-10 text-blue-400 animate-spin" />
                        <p className="text-sm text-slate-300">Processing & Indexing Video...</p>
                    </>
                ) : uploadStatus === 'success' ? (
                    <>
                        <CheckCircle className="w-10 h-10 text-green-400" />
                        <p className="text-sm text-green-400 font-medium">Indexing Complete! Ready to Chat.</p>
                        <span className="text-xs text-slate-400 mt-2">Click to upload a different video</span>
                    </>
                ) : (
                    <>
                        <Upload className="w-10 h-10 text-slate-400" />
                        <div>
                            <p className="text-sm font-medium text-slate-200">Click to upload video</p>
                            <p className="text-xs text-slate-500">MP4, MKV supported</p>
                        </div>
                    </>
                )}
            </label>
        </div>
    );
}