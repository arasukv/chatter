export interface DocumentChunkPreview {
  id: string;
  index: number;
  page?: number;
  text: string;
  charCount: number;
}

export interface IndexedDocumentSummary {
  documentId: string;
  fileName: string;
  fileType: "pdf" | "txt";
  fileSize: number;
  pageCount: number;
  chunkCount: number;
  characterCount: number;
  truncated: boolean;
  suggestedQuestions: string[];
  chunks: DocumentChunkPreview[];
}

export interface RetrievedSource {
  citationNumber: number;
  chunkId: string;
  chunkIndex: number;
  page?: number;
  score: number;
  text: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: RetrievedSource[];
  isStreaming?: boolean;
  error?: string;
}
