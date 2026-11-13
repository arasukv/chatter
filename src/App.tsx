import { useRef, useState } from "react";
import { RotateCcw } from "lucide-react";
import { DocumentUploader } from "./components/DocumentUploader";
import { ChatInterface } from "./components/ChatInterface";
import type {
  ChatMessage,
  IndexedDocumentSummary,
  RetrievedSource,
} from "./types/rag";

export default function App() {
  const [document, setDocument] = useState<IndexedDocumentSummary | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const abortControllerRef = useRef<AbortController | null>(null);

  const handleFileSelected = async (file: File) => {
    setUploadError(null);
    setIsUploading(true);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch("/api/documents", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || "Failed to index document.");
      }

      if (document?.documentId) {
        fetch(`/api/documents/${document.documentId}`, { method: "DELETE" }).catch(
          () => {}
        );
      }

      setDocument(data as IndexedDocumentSummary);
      setMessages([]);
    } catch (err) {
      setUploadError(
        err instanceof Error ? err.message : "Unable to process file."
      );
    } finally {
      setIsUploading(false);
    }
  };

  const handleLoadSample = async () => {
    setUploadError(null);
    setIsUploading(true);

    try {
      const response = await fetch("/api/documents/sample", {
        method: "POST",
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || "Failed to load sample document.");
      }

      if (document?.documentId) {
        fetch(`/api/documents/${document.documentId}`, { method: "DELETE" }).catch(
          () => {}
        );
      }

      setDocument(data as IndexedDocumentSummary);
      setMessages([]);
    } catch (err) {
      setUploadError(
        err instanceof Error ? err.message : "Unable to load sample document."
      );
    } finally {
      setIsUploading(false);
    }
  };

  const handleClearDocument = () => {
    abortControllerRef.current?.abort();
    setIsStreaming(false);

    if (document?.documentId) {
      fetch(`/api/documents/${document.documentId}`, { method: "DELETE" }).catch(
        () => {}
      );
    }
    setDocument(null);
    setMessages([]);
    setUploadError(null);
  };

  const handleResetChat = () => {
    abortControllerRef.current?.abort();
    setIsStreaming(false);
    setMessages([]);
  };

  const handleStopStreaming = () => {
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;
    setIsStreaming(false);
    setMessages((prev) =>
      prev.map((m) => (m.isStreaming ? { ...m, isStreaming: false } : m))
    );
  };

  const handleSendMessage = async (question: string) => {
    if (!document || isStreaming) return;

    const userMsg: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: question,
    };

    const assistantMsgId = crypto.randomUUID();
    const assistantMsg: ChatMessage = {
      id: assistantMsgId,
      role: "assistant",
      content: "",
      isStreaming: true,
    };

    const historySnapshot = messages
      .filter((m) => !m.error && m.content.trim().length > 0)
      .map((m) => ({ role: m.role, content: m.content }));

    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setIsStreaming(true);

    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          documentId: document.documentId,
          question,
          history: historySnapshot,
        }),
        signal: controller.signal,
      });

      if (!response.ok) {
        const errorJson = await response.json().catch(() => ({}));
        throw new Error(errorJson.error || "Failed to retrieve answer.");
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error("Streaming response is unavailable.");
      }

      const decoder = new TextDecoder();
      let buffer = "";
      let currentEvent = "message";

      const processLine = (line: string) => {
        if (line.startsWith("event:")) {
          currentEvent = line.slice(6).trim();
          return;
        }
        if (line.startsWith("data:")) {
          const rawData = line.slice(5).trim();
          if (!rawData) return;

          const parsed = JSON.parse(rawData) as
            | RetrievedSource[]
            | { text?: string; message?: string };

          if (currentEvent === "sources" && Array.isArray(parsed)) {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantMsgId ? { ...m, sources: parsed } : m
              )
            );
          } else if (
            currentEvent === "delta" &&
            !Array.isArray(parsed) &&
            typeof parsed.text === "string"
          ) {
            const deltaText = parsed.text;
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantMsgId
                  ? { ...m, content: m.content + deltaText }
                  : m
              )
            );
          } else if (
            currentEvent === "error" &&
            !Array.isArray(parsed) &&
            parsed.message
          ) {
            const errMessage = parsed.message;
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantMsgId
                  ? { ...m, error: errMessage, isStreaming: false }
                  : m
              )
            );
          }
        }
      };

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const rawLine of lines) {
          const line = rawLine.trimEnd();
          if (line === "") {
            currentEvent = "message";
            continue;
          }
          processLine(line);
        }
      }

      if (buffer.trim()) {
        processLine(buffer.trimEnd());
      }

      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantMsgId ? { ...m, isStreaming: false } : m
        )
      );
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMsgId ? { ...m, isStreaming: false } : m
          )
        );
        return;
      }

      const errorMessage =
        err instanceof Error ? err.message : "Streaming request failed.";
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantMsgId
            ? { ...m, error: errorMessage, isStreaming: false }
            : m
        )
      );
    } finally {
      abortControllerRef.current = null;
      setIsStreaming(false);
    }
  };

  return (
    <div className="min-h-screen w-full bg-[#F5F6F8] px-4 py-6 sm:px-6 sm:py-10 flex items-center justify-center">
      <main className="w-full max-w-[920px] h-[min(860px,calc(100vh-3rem))] min-h-[600px] flex flex-col rounded-xl border border-slate-200 bg-white shadow-[0_1px_3px_rgba(15,23,42,0.06)] overflow-hidden">
        <header className="flex items-center justify-between gap-6 border-b border-slate-200 px-5 py-4 sm:px-6">
          <div className="flex items-baseline gap-3 min-w-0">
            <h1 className="text-lg font-bold tracking-tight text-slate-900 whitespace-nowrap shrink-0">
              Chatter
            </h1>
            <span className="hidden sm:inline text-xs text-slate-500 truncate">
              Document Q&amp;A with in-memory cosine retrieval
            </span>
          </div>

          {messages.length > 0 && (
            <button
              type="button"
              onClick={handleResetChat}
              className="inline-flex items-center gap-1.5 rounded-md border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-50 hover:text-slate-900 transition-colors whitespace-nowrap shrink-0 cursor-pointer"
            >
              <RotateCcw className="h-3.5 w-3.5" />
              <span>Clear chat</span>
            </button>
          )}
        </header>

        <DocumentUploader
          document={document}
          isUploading={isUploading}
          uploadError={uploadError}
          onFileSelected={handleFileSelected}
          onLoadSample={handleLoadSample}
          onClearDocument={handleClearDocument}
        />

        <ChatInterface
          document={document}
          messages={messages}
          isStreaming={isStreaming}
          onSendMessage={handleSendMessage}
          onStopStreaming={handleStopStreaming}
        />
      </main>
    </div>
  );
}
