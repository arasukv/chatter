import {
  useEffect,
  useRef,
  useState,
  type FormEvent,
  type KeyboardEvent,
} from "react";
import Markdown from "react-markdown";
import {
  ArrowUp,
  Square,
  ChevronDown,
  ChevronUp,
  ArrowRight,
  AlertCircle,
} from "lucide-react";
import type {
  ChatMessage,
  IndexedDocumentSummary,
  RetrievedSource,
} from "../types/rag";

interface ChatInterfaceProps {
  document: IndexedDocumentSummary | null;
  messages: ChatMessage[];
  isStreaming: boolean;
  onSendMessage: (question: string) => void;
  onStopStreaming: () => void;
}

function SourcesDisclosure({ sources }: { sources: RetrievedSource[] }) {
  const [expanded, setExpanded] = useState(false);

  if (sources.length === 0) return null;

  const topScore = sources[0]?.score ?? 0;

  return (
    <div className="mt-3 pt-2.5 border-t border-slate-100">
      <button
        type="button"
        onClick={() => setExpanded((prev) => !prev)}
        className="inline-flex items-center gap-1.5 text-xs text-slate-500 hover:text-slate-800 transition-colors cursor-pointer"
      >
        <span className="font-medium">
          {sources.length} retrieved {sources.length === 1 ? "passage" : "passages"}
        </span>
        <span aria-hidden="true">·</span>
        <span className="font-mono tabular-nums">
          top cosine {topScore.toFixed(3)}
        </span>
        {expanded ? (
          <ChevronUp className="h-3.5 w-3.5" />
        ) : (
          <ChevronDown className="h-3.5 w-3.5" />
        )}
      </button>

      {expanded && (
        <div className="mt-2.5 divide-y divide-slate-100 rounded-lg border border-slate-200 bg-slate-50/60">
          {sources.map((src) => (
            <div key={src.chunkId} className="px-3.5 py-2.5 text-xs">
              <div className="mb-1 flex items-center gap-2 text-slate-500 font-mono tabular-nums">
                <span className="font-semibold text-slate-800">
                  [{src.citationNumber}]
                </span>
                {src.page && (
                  <>
                    <span aria-hidden="true">·</span>
                    <span>Page {src.page}</span>
                  </>
                )}
                <span aria-hidden="true">·</span>
                <span>Chunk {src.chunkIndex}</span>
                <span aria-hidden="true">·</span>
                <span>cosine {src.score.toFixed(3)}</span>
              </div>
              <p className="text-slate-600 leading-relaxed whitespace-pre-wrap">
                {src.text}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export function ChatInterface({
  document,
  messages,
  isStreaming,
  onSendMessage,
  onStopStreaming,
}: ChatInterfaceProps) {
  const [input, setInput] = useState("");
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const el = scrollContainerRef.current;
    if (el) {
      el.scrollTop = el.scrollHeight;
    }
  }, [messages]);

  const adjustTextareaHeight = () => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 128)}px`;
  };

  const submitQuestion = () => {
    const trimmed = input.trim();
    if (!trimmed || !document || isStreaming) return;
    setInput("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
    onSendMessage(trimmed);
  };

  const handleFormSubmit = (e: FormEvent) => {
    e.preventDefault();
    submitQuestion();
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submitQuestion();
    }
  };

  return (
    <div className="flex flex-1 flex-col min-h-0">
      <div
        ref={scrollContainerRef}
        className="flex-1 overflow-y-auto px-5 py-6 sm:px-6"
      >
        {!document ? (
          <div className="flex h-full min-h-[280px] flex-col items-center justify-center text-center px-4">
            <p className="text-sm font-medium text-slate-700">
              No document loaded yet
            </p>
            <p className="mt-1 max-w-md text-xs text-slate-500 leading-relaxed">
              Upload a PDF or plain-text document above. Your questions will be
              answered strictly from the top cosine-matched passages in memory.
            </p>
          </div>
        ) : messages.length === 0 ? (
          <div className="flex h-full min-h-[280px] flex-col justify-center max-w-xl mx-auto">
            <p className="text-sm font-medium text-slate-900">
              Document ready for questions
            </p>
            <p className="mt-1 text-xs text-slate-500">
              Ask anything about{" "}
              <span className="font-medium text-slate-700">
                {document.fileName}
              </span>
              , or start with one of these prompts:
            </p>

            <div className="mt-4 divide-y divide-slate-100 rounded-lg border border-slate-200 bg-white">
              {document.suggestedQuestions.map((question) => (
                <button
                  key={question}
                  type="button"
                  onClick={() => onSendMessage(question)}
                  className="group flex w-full items-center justify-between gap-3 px-4 py-3 text-left text-sm text-slate-700 hover:bg-slate-50 hover:text-slate-900 transition-colors cursor-pointer"
                >
                  <span>{question}</span>
                  <ArrowRight className="h-3.5 w-3.5 shrink-0 text-slate-400 group-hover:text-slate-700 group-hover:translate-x-0.5 transition-all" />
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="space-y-6 max-w-3xl mx-auto">
            {messages.map((msg) => (
              <div key={msg.id}>
                {msg.role === "user" ? (
                  <div className="flex justify-end">
                    <div className="max-w-[82%] rounded-lg bg-slate-100 px-4 py-2.5 text-sm text-slate-900 leading-relaxed whitespace-pre-wrap">
                      {msg.content}
                    </div>
                  </div>
                ) : (
                  <div className="space-y-1">
                    {msg.error ? (
                      <div
                        role="alert"
                        className="flex items-start gap-2 rounded-lg border border-red-200 bg-red-50 px-3.5 py-2.5 text-xs text-red-700"
                      >
                        <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" />
                        <span>{msg.error}</span>
                      </div>
                    ) : (
                      <div className="text-[15px] text-slate-800 leading-relaxed">
                        {msg.content ? (
                          <div className="space-y-3 [&_p]:leading-relaxed [&_ul]:list-disc [&_ul]:pl-5 [&_ul]:space-y-1 [&_ol]:list-decimal [&_ol]:pl-5 [&_ol]:space-y-1 [&_code]:rounded [&_code]:bg-slate-100 [&_code]:px-1.5 [&_code]:py-0.5 [&_code]:font-mono [&_code]:text-[13px] [&_strong]:font-semibold [&_strong]:text-slate-900">
                            <Markdown>{msg.content}</Markdown>
                          </div>
                        ) : msg.isStreaming ? (
                          <span className="inline-flex items-center gap-1.5 text-xs text-slate-500">
                            <span className="inline-block h-1.5 w-1.5 rounded-full bg-slate-400 animate-pulse" />
                            <span>Retrieving passages and drafting answer...</span>
                          </span>
                        ) : null}

                        {msg.isStreaming && msg.content && (
                          <span
                            aria-hidden="true"
                            className="ml-1 inline-block h-3.5 w-1.5 align-middle bg-slate-900 animate-pulse"
                          />
                        )}
                      </div>
                    )}

                    {msg.sources && msg.sources.length > 0 && !msg.error && (
                      <SourcesDisclosure sources={msg.sources} />
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      <form
        onSubmit={handleFormSubmit}
        className="border-t border-slate-200 bg-white px-4 py-3.5 sm:px-6"
      >
        <div className="flex items-end gap-2.5 rounded-lg border border-slate-200 bg-white px-3.5 py-2 focus-within:border-blue-600 focus-within:ring-1 focus-within:ring-blue-600 transition-shadow">
          <textarea
            ref={textareaRef}
            rows={1}
            value={input}
            onChange={(e) => {
              setInput(e.target.value);
              adjustTextareaHeight();
            }}
            onKeyDown={handleKeyDown}
            disabled={!document}
            placeholder={
              document
                ? `Ask a question about ${document.fileName}...`
                : "Upload a PDF or TXT document above to begin..."
            }
            className="flex-1 resize-none bg-transparent py-1 text-sm text-slate-900 placeholder:text-slate-400 focus:outline-none disabled:cursor-not-allowed disabled:text-slate-400"
          />

          {isStreaming ? (
            <button
              type="button"
              onClick={onStopStreaming}
              aria-label="Stop generating"
              className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-slate-900 text-white hover:bg-slate-800 transition-colors cursor-pointer"
            >
              <Square className="h-3.5 w-3.5 fill-current" />
            </button>
          ) : (
            <button
              type="submit"
              disabled={!document || !input.trim()}
              aria-label="Send question"
              className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-slate-900 text-white hover:bg-slate-800 disabled:bg-slate-100 disabled:text-slate-400 disabled:cursor-not-allowed transition-colors cursor-pointer"
            >
              <ArrowUp className="h-4 w-4" />
            </button>
          )}
        </div>
        <div className="mt-2 flex items-center justify-between text-[11px] text-slate-400">
          <span>Answers are grounded strictly in retrieved document chunks.</span>
          <span className="hidden sm:inline">Enter to send · Shift+Enter for new line</span>
        </div>
      </form>
    </div>
  );
}
