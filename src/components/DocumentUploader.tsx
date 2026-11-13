import { useRef, useState, type DragEvent, type ChangeEvent } from "react";
import {
  FileText,
  Upload,
  X,
  ChevronDown,
  ChevronUp,
  RefreshCw,
  AlertCircle,
} from "lucide-react";
import type { IndexedDocumentSummary } from "../types/rag";

interface DocumentUploaderProps {
  document: IndexedDocumentSummary | null;
  isUploading: boolean;
  uploadError: string | null;
  onFileSelected: (file: File) => void;
  onLoadSample: () => void;
  onClearDocument: () => void;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

export function DocumentUploader({
  document,
  isUploading,
  uploadError,
  onFileSelected,
  onLoadSample,
  onClearDocument,
}: DocumentUploaderProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [showChunks, setShowChunks] = useState(false);

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (!isUploading) {
      setIsDragging(true);
    }
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    if (isUploading) return;

    const file = e.dataTransfer.files?.[0];
    if (file) {
      onFileSelected(file);
    }
  };

  const handleInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onFileSelected(file);
    }
    e.target.value = "";
  };

  const triggerBrowse = () => {
    if (!isUploading) {
      fileInputRef.current?.click();
    }
  };

  return (
    <div
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      className="border-b border-slate-200 bg-slate-50/60 transition-colors"
    >
      <input
        ref={fileInputRef}
        type="file"
        accept=".pdf,.txt,.md,.csv,.json,.log,application/pdf,text/plain"
        onChange={handleInputChange}
        className="hidden"
        aria-label="Upload PDF or text document"
      />

      {!document ? (
        <div className="p-5 sm:p-6">
          <div
            onClick={triggerBrowse}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                triggerBrowse();
              }
            }}
            className={`group relative flex flex-col items-center justify-center rounded-lg border border-dashed px-6 py-7 text-center transition-all cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-600 focus-visible:ring-offset-2 ${
              isDragging
                ? "border-blue-600 bg-blue-50/60"
                : "border-slate-300 bg-white hover:border-slate-400 hover:bg-slate-50/50"
            } ${isUploading ? "pointer-events-none opacity-80" : ""}`}
          >
            {isUploading ? (
              <div className="flex flex-col items-center gap-3 w-full max-w-sm">
                <div className="h-9 w-9 rounded-lg bg-slate-100 flex items-center justify-center text-slate-700">
                  <RefreshCw className="h-4 w-4 animate-spin" />
                </div>
                <div className="space-y-1">
                  <p className="text-sm font-medium text-slate-900">
                    Indexing document in memory
                  </p>
                  <p className="text-xs text-slate-500">
                    Extracting text, chunking passages, and computing embeddings
                  </p>
                </div>
                <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-100">
                  <div className="h-full w-1/2 animate-pulse rounded-full bg-slate-900" />
                </div>
              </div>
            ) : (
              <>
                <div className="mb-3 flex h-10 w-10 items-center justify-center rounded-lg bg-slate-100 text-slate-700 group-hover:bg-slate-200/70 transition-colors">
                  <Upload className="h-4 w-4" />
                </div>
                <p className="text-sm font-medium text-slate-900">
                  Drop a PDF or TXT file here, or{" "}
                  <span className="text-blue-600 underline decoration-blue-600/30 underline-offset-2 group-hover:decoration-blue-600">
                    browse
                  </span>
                </p>
                <p className="mt-1 text-xs text-slate-500">
                  Chunked and embedded in memory for cosine-similarity retrieval · Up to 15 MB
                </p>
              </>
            )}
          </div>

          {!isUploading && (
            <div className="mt-3 flex items-center justify-between gap-4 text-xs text-slate-500">
              <span>Supported formats: PDF, TXT, MD, CSV</span>
              <button
                type="button"
                onClick={onLoadSample}
                className="font-medium text-slate-700 hover:text-slate-900 underline decoration-slate-300 underline-offset-2 hover:decoration-slate-600 transition-colors whitespace-nowrap shrink-0 cursor-pointer"
              >
                Load sample technical guide
              </button>
            </div>
          )}

          {uploadError && (
            <div
              role="alert"
              className="mt-3 flex items-start gap-2 rounded-lg border border-red-200 bg-red-50 px-3.5 py-2.5 text-xs text-red-700"
            >
              <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" />
              <span>{uploadError}</span>
            </div>
          )}
        </div>
      ) : (
        <div>
          <div
            className={`flex flex-wrap items-center justify-between gap-3 px-5 py-3.5 sm:px-6 transition-colors ${
              isDragging ? "bg-blue-50/70" : ""
            }`}
          >
            <div className="flex items-center gap-3 min-w-0">
              <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border border-slate-200 bg-white text-slate-700">
                <FileText className="h-4 w-4" />
              </div>
              <div className="min-w-0">
                <p className="truncate text-sm font-semibold text-slate-900 max-w-[260px] sm:max-w-[380px]">
                  {document.fileName}
                </p>
                <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 text-xs text-slate-500 font-mono tabular-nums">
                  <span>{document.fileType.toUpperCase()}</span>
                  {document.fileType === "pdf" && (
                    <>
                      <span aria-hidden="true">·</span>
                      <span>
                        {document.pageCount}{" "}
                        {document.pageCount === 1 ? "page" : "pages"}
                      </span>
                    </>
                  )}
                  <span aria-hidden="true">·</span>
                  <span>{document.chunkCount} chunks</span>
                  <span aria-hidden="true">·</span>
                  <span>{formatBytes(document.fileSize)}</span>
                  {document.truncated && (
                    <>
                      <span aria-hidden="true">·</span>
                      <span className="text-amber-700 font-sans">
                        Capped at {document.chunkCount} chunks
                      </span>
                    </>
                  )}
                </div>
              </div>
            </div>

            <div className="flex items-center gap-2 shrink-0">
              <button
                type="button"
                onClick={() => setShowChunks((prev) => !prev)}
                className="inline-flex items-center gap-1.5 rounded-md border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-50 hover:text-slate-900 transition-colors whitespace-nowrap shrink-0 cursor-pointer"
              >
                <span>{showChunks ? "Hide chunks" : "Inspect chunks"}</span>
                {showChunks ? (
                  <ChevronUp className="h-3.5 w-3.5" />
                ) : (
                  <ChevronDown className="h-3.5 w-3.5" />
                )}
              </button>
              <button
                type="button"
                onClick={triggerBrowse}
                disabled={isUploading}
                className="inline-flex items-center gap-1.5 rounded-md border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-50 hover:text-slate-900 disabled:opacity-50 transition-colors whitespace-nowrap shrink-0 cursor-pointer"
              >
                {isUploading ? (
                  <RefreshCw className="h-3.5 w-3.5 animate-spin" />
                ) : (
                  <Upload className="h-3.5 w-3.5" />
                )}
                <span>{isUploading ? "Indexing..." : "Replace"}</span>
              </button>
              <button
                type="button"
                onClick={() => {
                  setShowChunks(false);
                  onClearDocument();
                }}
                disabled={isUploading}
                aria-label="Remove document"
                className="inline-flex items-center justify-center rounded-md border border-slate-200 bg-white p-1.5 text-slate-500 hover:bg-slate-50 hover:text-slate-900 disabled:opacity-50 transition-colors cursor-pointer"
              >
                <X className="h-3.5 w-3.5" />
              </button>
            </div>
          </div>

          {uploadError && (
            <div
              role="alert"
              className="mx-5 mb-3 sm:mx-6 flex items-start gap-2 rounded-lg border border-red-200 bg-red-50 px-3.5 py-2.5 text-xs text-red-700"
            >
              <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" />
              <span>{uploadError}</span>
            </div>
          )}

          {showChunks && (
            <div className="border-t border-slate-200 bg-white px-5 py-3.5 sm:px-6">
              <div className="mb-2.5 flex items-center justify-between text-xs text-slate-500">
                <span className="font-medium text-slate-700">
                  In-memory document chunks
                </span>
                <span className="font-mono tabular-nums">
                  {document.characterCount.toLocaleString()} chars total
                </span>
              </div>
              <div className="max-h-52 overflow-y-auto divide-y divide-slate-100 border border-slate-200 rounded-lg">
                {document.chunks.map((chunk) => (
                  <div key={chunk.id} className="px-3.5 py-2.5 text-xs">
                    <div className="mb-1 flex items-center gap-2 text-slate-500 font-mono tabular-nums">
                      <span className="font-medium text-slate-700">
                        Chunk {chunk.index}
                      </span>
                      {chunk.page && (
                        <>
                          <span aria-hidden="true">·</span>
                          <span>Page {chunk.page}</span>
                        </>
                      )}
                      <span aria-hidden="true">·</span>
                      <span>{chunk.charCount} chars</span>
                    </div>
                    <p className="text-slate-600 leading-relaxed line-clamp-3">
                      {chunk.text}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
