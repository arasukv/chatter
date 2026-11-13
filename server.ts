import "dotenv/config";
import crypto from "node:crypto";
import path from "node:path";
import express from "express";
import multer from "multer";
import { extractText } from "unpdf";
import { GoogleGenAI, ThinkingLevel, Type } from "@google/genai";
import type {
  DocumentChunkPreview,
  IndexedDocumentSummary,
  RetrievedSource,
} from "./src/types/rag.ts";

interface StoredChunk {
  id: string;
  index: number;
  page?: number;
  text: string;
  charCount: number;
  embedding: number[];
}

interface StoredDocument {
  documentId: string;
  fileName: string;
  fileType: "pdf" | "txt";
  fileSize: number;
  pageCount: number;
  characterCount: number;
  truncated: boolean;
  suggestedQuestions: string[];
  chunks: StoredChunk[];
  createdAt: number;
}

const MAX_FILE_BYTES = 15 * 1024 * 1024; // 15 MB
const MAX_CHUNKS = 240;
const CHUNK_SIZE = 850;
const CHUNK_OVERLAP = 150;
const EMBED_BATCH_SIZE = 24;
const TOP_K_RESULTS = 5;
const MAX_STORED_DOCS = 20;

const documentStore = new Map<string, StoredDocument>();

function getGeminiClient(): GoogleGenAI {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error(
      "GEMINI_API_KEY is missing. Please configure it in the Secrets panel."
    );
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        "User-Agent": "aistudio-build",
      },
    },
  });
}

function formatGeminiError(err: unknown): string {
  const message = err instanceof Error ? err.message : String(err);
  if (message.includes("API_KEY_INVALID") || message.includes("PERMISSION_DENIED") || message.includes("403")) {
    return "Gemini API authentication failed. Check your GEMINI_API_KEY in Settings > Secrets.";
  }
  if (message.includes("RESOURCE_EXHAUSTED") || message.includes("429")) {
    return "Gemini API rate limit reached. Please wait a moment and try again.";
  }
  if (message.includes("NOT_FOUND") || message.includes("404")) {
    return "Requested Gemini model was not found or is unavailable.";
  }
  return message || "An unexpected error occurred while processing the request.";
}

async function withRetry<T>(fn: () => Promise<T>, attempts = 3): Promise<T> {
  let lastError: unknown;
  for (let i = 0; i < attempts; i++) {
    try {
      return await fn();
    } catch (err) {
      lastError = err;
      const msg = err instanceof Error ? err.message : String(err);
      const isRetryable =
        msg.includes("429") ||
        msg.includes("503") ||
        msg.includes("RESOURCE_EXHAUSTED") ||
        msg.includes("UNAVAILABLE");
      if (!isRetryable || i === attempts - 1) {
        throw err;
      }
      const delayMs = 450 * Math.pow(2, i);
      await new Promise((resolve) => setTimeout(resolve, delayMs));
    }
  }
  throw lastError;
}

function normalizeText(raw: string): string {
  return raw
    .replace(/\r\n/g, "\n")
    .replace(/\t/g, " ")
    .replace(/[ ]{2,}/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function splitSegmentIntoChunks(
  text: string,
  pageNumber?: number
): Array<{ text: string; page?: number }> {
  const cleaned = normalizeText(text);
  if (!cleaned) return [];

  if (cleaned.length <= CHUNK_SIZE) {
    return [{ text: cleaned, page: pageNumber }];
  }

  const results: Array<{ text: string; page?: number }> = [];
  let start = 0;

  while (start < cleaned.length) {
    let end = Math.min(start + CHUNK_SIZE, cleaned.length);

    if (end < cleaned.length) {
      const windowSlice = cleaned.slice(start, end);
      const paragraphBreak = windowSlice.lastIndexOf("\n\n");
      const sentenceBreak = Math.max(
        windowSlice.lastIndexOf(". "),
        windowSlice.lastIndexOf("? "),
        windowSlice.lastIndexOf("! "),
        windowSlice.lastIndexOf(".\n")
      );
      const spaceBreak = windowSlice.lastIndexOf(" ");

      if (paragraphBreak > CHUNK_SIZE * 0.45) {
        end = start + paragraphBreak + 2;
      } else if (sentenceBreak > CHUNK_SIZE * 0.45) {
        end = start + sentenceBreak + 1;
      } else if (spaceBreak > CHUNK_SIZE * 0.3) {
        end = start + spaceBreak;
      }
    }

    const slice = cleaned.slice(start, end).trim();
    if (slice.length > 0) {
      results.push({ text: slice, page: pageNumber });
    }

    if (end >= cleaned.length) {
      break;
    }

    const nextStart = Math.max(start + 1, end - CHUNK_OVERLAP);
    start = nextStart;
  }

  return results;
}

function chunkPages(pages: Array<{ text: string; page?: number }>): {
  chunks: Array<{ id: string; index: number; page?: number; text: string; charCount: number }>;
  truncated: boolean;
  totalChars: number;
} {
  const rawChunks: Array<{ text: string; page?: number }> = [];
  let totalChars = 0;

  for (const pageEntry of pages) {
    const cleaned = normalizeText(pageEntry.text);
    if (!cleaned) continue;
    totalChars += cleaned.length;
    const pageChunks = splitSegmentIntoChunks(cleaned, pageEntry.page);
    rawChunks.push(...pageChunks);
  }

  // Merge tiny trailing chunks if helpful
  const merged: Array<{ text: string; page?: number }> = [];
  for (const item of rawChunks) {
    if (
      merged.length > 0 &&
      item.text.length < 120 &&
      merged[merged.length - 1].text.length + item.text.length + 1 <= CHUNK_SIZE + 120
    ) {
      merged[merged.length - 1].text = `${merged[merged.length - 1].text}\n\n${item.text}`;
    } else {
      merged.push(item);
    }
  }

  const truncated = merged.length > MAX_CHUNKS;
  const bounded = merged.slice(0, MAX_CHUNKS);

  return {
    chunks: bounded.map((c, idx) => ({
      id: `chunk-${idx + 1}`,
      index: idx + 1,
      page: c.page,
      text: c.text,
      charCount: c.text.length,
    })),
    truncated,
    totalChars,
  };
}

async function embedTexts(ai: GoogleGenAI, texts: string[]): Promise<number[][]> {
  const vectors: number[][] = [];

  for (let i = 0; i < texts.length; i += EMBED_BATCH_SIZE) {
    const batch = texts.slice(i, i + EMBED_BATCH_SIZE);
    const response = await withRetry(() =>
      ai.models.embedContent({
        model: "gemini-embedding-2",
        contents: batch,
      })
    );

    const embeddings = response.embeddings ?? [];
    if (embeddings.length !== batch.length) {
      throw new Error("Embedding service returned an incomplete batch.");
    }

    for (const item of embeddings) {
      if (!item.values || item.values.length === 0) {
        throw new Error("Embedding service returned an empty vector.");
      }
      vectors.push(item.values);
    }
  }

  return vectors;
}

async function generateSuggestedQuestions(
  ai: GoogleGenAI,
  fileName: string,
  sampleChunks: string[]
): Promise<string[]> {
  try {
    const preview = sampleChunks.slice(0, 4).join("\n\n---\n\n");
    const response = await ai.models.generateContent({
      model: "gemini-4-hydrogen",
      contents: `Document filename: ${fileName}\n\nExcerpts:\n${preview}\n\nGenerate 3 concise, specific questions a reader could ask that are directly answerable from this document.`,
      config: {
        systemInstruction:
          "Return strictly 3 short, natural questions (under 14 words each) grounded in the provided document excerpts.",
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.ARRAY,
          items: { type: Type.STRING },
        },
        thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
      },
    });
    const parsed = JSON.parse((response.text ?? "[]").trim()) as unknown;
    if (Array.isArray(parsed)) {
      return parsed
        .filter((q): q is string => typeof q === "string" && q.trim().length > 0)
        .slice(0, 3);
    }
  } catch {
    // Non-fatal fallback
  }
  return [
    "What are the main topics covered in this document?",
    "Summarize the key findings or requirements.",
    "Are there any specific metrics, dates, or deadlines mentioned?",
  ];
}

function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length || a.length === 0) return 0;
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    const va = a[i];
    const vb = b[i];
    dot += va * vb;
    normA += va * va;
    normB += vb * vb;
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

function pruneOldDocuments() {
  if (documentStore.size <= MAX_STORED_DOCS) return;
  const sorted = Array.from(documentStore.values()).sort(
    (a, b) => a.createdAt - b.createdAt
  );
  const toDelete = sorted.length - MAX_STORED_DOCS;
  for (let i = 0; i < toDelete; i++) {
    documentStore.delete(sorted[i].documentId);
  }
}

async function indexDocumentInMemory(params: {
  fileName: string;
  fileType: "pdf" | "txt";
  fileSize: number;
  pages: Array<{ text: string; page?: number }>;
}): Promise<IndexedDocumentSummary> {
  const { chunks, truncated, totalChars } = chunkPages(params.pages);
  if (chunks.length === 0) {
    throw new Error(
      "No readable text could be extracted from this file. If this is a scanned PDF without embedded text, please upload a text-based PDF or TXT file."
    );
  }

  const ai = getGeminiClient();
  const vectors = await embedTexts(
    ai,
    chunks.map((c) => c.text)
  );

  const storedChunks: StoredChunk[] = chunks.map((chunk, idx) => ({
    ...chunk,
    embedding: vectors[idx],
  }));

  const suggestedQuestions = await generateSuggestedQuestions(
    ai,
    params.fileName,
    storedChunks.map((c) => c.text)
  );

  const documentId = crypto.randomUUID();
  const storedDoc: StoredDocument = {
    documentId,
    fileName: params.fileName,
    fileType: params.fileType,
    fileSize: params.fileSize,
    pageCount: params.fileType === "pdf" ? params.pages.length : 1,
    characterCount: totalChars,
    truncated,
    suggestedQuestions,
    chunks: storedChunks,
    createdAt: Date.now(),
  };

  documentStore.set(documentId, storedDoc);
  pruneOldDocuments();

  const previewChunks: DocumentChunkPreview[] = storedChunks.map((c) => ({
    id: c.id,
    index: c.index,
    page: c.page,
    text: c.text,
    charCount: c.charCount,
  }));

  return {
    documentId: storedDoc.documentId,
    fileName: storedDoc.fileName,
    fileType: storedDoc.fileType,
    fileSize: storedDoc.fileSize,
    pageCount: storedDoc.pageCount,
    chunkCount: storedDoc.chunks.length,
    characterCount: storedDoc.characterCount,
    truncated: storedDoc.truncated,
    suggestedQuestions: storedDoc.suggestedQuestions,
    chunks: previewChunks,
  };
}

const SAMPLE_DOCUMENT_TEXT = `Northstar Field Telemetry & Battery Operations Guide (Revision 4.2)

1. System Overview & Deployment Architecture
Northstar Edge Nodes are solar-assisted environmental telemetry stations designed for remote watershed and alpine monitoring. Each node captures barometric pressure, dissolved oxygen, water turbidity, and ambient temperature at 60-second sampling intervals. Data packets are buffered locally in 64 MB non-volatile flash memory and transmitted over LoRaWAN (915 MHz ISM band in North America, 868 MHz in Europe) every 15 minutes to the nearest cellular backhaul gateway.

In regions with dense canopy cover, operators must configure the spreading factor to SF10 or SF11 to preserve link margin above 12 dB, recognizing that higher spreading factors reduce payload throughput from 5.4 kbps to 980 bps and increase radio transmit duty cycle by 38%.

2. Power Budget & Cold-Weather Battery Management
Each Northstar Edge Node is powered by a 12.8V 18Ah LiFePO4 battery pack paired with a 30W monocrystalline solar panel and an MPPT charge controller. Under standard 15-minute transmission cadences, average continuous draw is 42 mW in standby and 680 mW during active LoRaWAN uplink bursts.

Critical Cold-Weather Rule: LiFePO4 cells must never be charged below 0°C (32°F) due to permanent lithium plating risk on the anode. When the internal thermistor reads below 1.5°C, the firmware automatically opens the charge-enable FET while continuing to allow discharge down to -20°C. Operators deploying stations above 2,400 meters elevation must enable the insulated thermal sleeve option (Part #NS-THRM-09), which diverts 15% of daytime solar surplus to a self-regulating 4W polyimide heater pad before initiating battery charge.

3. Calibration Schedule & Sensor Maintenance
Dissolved oxygen optical caps drift at a predictable rate of 0.12 mg/L per quarter due to biofilm accumulation and fluorophore photobleaching. Field technicians must perform a two-point wet calibration (0% sodium sulfite zero-oxygen bath and 100% water-saturated air) every 180 days. Optical sensing caps must be replaced after 24 months of continuous submersion regardless of passing calibration checks.

Turbidity wipers operate automatically before every 15-minute reading cycle. If wiper current draw exceeds 310 mA for three consecutive sweeps, the node flags fault code E-402 ("Wiper Obstruction") and suspends mechanical wiping to avoid stripping the nylon drive gear while continuing optical measurements.

4. Incident Response & Firmware Recovery
If a gateway loses cellular backhaul for more than 4 hours, connected Edge Nodes switch into Store-and-Forward mode. Up to 14 days of 60-second telemetry can be retained locally before circular buffer rollover overwrites the oldest records. Upon link restoration, nodes replay buffered frames at a throttled rate of 4 historical packets per uplink window so live telemetry is never starved.

To force a factory bootloader recovery in the field, hold the magnetic reed switch at the base of the enclosure for exactly 8 seconds until the diagnostic LED flashes amber three times, then connect via USB-C serial console at 115200 baud.`;

async function createApp() {
  const app = express();
  app.use(express.json({ limit: "2mb" }));

  const upload = multer({
    storage: multer.memoryStorage(),
    limits: { fileSize: MAX_FILE_BYTES },
  });

  app.get("/api/health", (_req, res) => {
    res.json({ ok: true });
  });

  app.post("/api/documents", (req, res) => {
    upload.single("file")(req, res, async (uploadErr) => {
      if (uploadErr) {
        if (uploadErr instanceof multer.MulterError && uploadErr.code === "LIMIT_FILE_SIZE") {
          return res.status(400).json({
            error: "File exceeds the 15 MB limit. Please upload a smaller PDF or text file.",
          });
        }
        return res.status(400).json({ error: uploadErr.message || "File upload failed." });
      }

      const file = req.file;
      if (!file) {
        return res.status(400).json({ error: "No file was provided." });
      }

      const ext = path.extname(file.originalname).toLowerCase();
      const isPdf = ext === ".pdf" || file.mimetype === "application/pdf";
      const isText =
        ext === ".txt" ||
        ext === ".md" ||
        ext === ".csv" ||
        ext === ".json" ||
        ext === ".log" ||
        file.mimetype.startsWith("text/");

      if (!isPdf && !isText) {
        return res.status(400).json({
          error: "Unsupported file format. Please upload a .pdf or .txt file.",
        });
      }

      try {
        let pages: Array<{ text: string; page?: number }> = [];
        if (isPdf) {
          const extracted = await extractText(new Uint8Array(file.buffer), {
            mergePages: false,
          });
          const rawPages = Array.isArray(extracted.text)
            ? extracted.text
            : [String(extracted.text ?? "")];
          pages = rawPages.map((pageText, index) => ({
            text: pageText,
            page: index + 1,
          }));
        } else {
          pages = [{ text: file.buffer.toString("utf-8") }];
        }

        const summary = await indexDocumentInMemory({
          fileName: file.originalname,
          fileType: isPdf ? "pdf" : "txt",
          fileSize: file.size,
          pages,
        });

        return res.json(summary);
      } catch (err) {
        console.error("Document indexing error:", err);
        return res.status(500).json({ error: formatGeminiError(err) });
      }
    });
  });

  app.post("/api/documents/sample", async (_req, res) => {
    try {
      const buffer = Buffer.from(SAMPLE_DOCUMENT_TEXT, "utf-8");
      const summary = await indexDocumentInMemory({
        fileName: "northstar-telemetry-guide.txt",
        fileType: "txt",
        fileSize: buffer.byteLength,
        pages: [{ text: SAMPLE_DOCUMENT_TEXT }],
      });
      return res.json(summary);
    } catch (err) {
      console.error("Sample document indexing error:", err);
      return res.status(500).json({ error: formatGeminiError(err) });
    }
  });

  app.delete("/api/documents/:id", (req, res) => {
    documentStore.delete(req.params.id);
    res.json({ deleted: true });
  });

  app.post("/api/chat", async (req, res) => {
    const { documentId, question, history } = req.body as {
      documentId?: string;
      question?: string;
      history?: Array<{ role: "user" | "assistant"; content: string }>;
    };

    if (!documentId || !question || !question.trim()) {
      return res.status(400).json({
        error: "Both documentId and a non-empty question are required.",
      });
    }

    const doc = documentStore.get(documentId);
    if (!doc) {
      return res.status(404).json({
        error:
          "This document session has expired from memory. Please re-upload your PDF or TXT file.",
      });
    }

    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache, no-transform");
    res.setHeader("Connection", "keep-alive");
    res.flushHeaders?.();

    const sendEvent = (event: string, data: unknown) => {
      if (!res.writableEnded) {
        res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
      }
    };

    try {
      const ai = getGeminiClient();

      // Embed the user question (optionally enriched with latest user context for short follow-ups)
      const [queryVector] = await embedTexts(ai, [question.trim()]);

      // Cosine similarity scoring across all in-memory chunks
      const scored = doc.chunks
        .map((chunk) => ({
          chunk,
          score: cosineSimilarity(queryVector, chunk.embedding),
        }))
        .sort((a, b) => b.score - a.score);

      const topMatches = scored.slice(0, Math.min(TOP_K_RESULTS, scored.length));

      const retrievedSources: RetrievedSource[] = topMatches.map((match, idx) => ({
        citationNumber: idx + 1,
        chunkId: match.chunk.id,
        chunkIndex: match.chunk.index,
        page: match.chunk.page,
        score: Number(match.score.toFixed(4)),
        text: match.chunk.text,
      }));

      sendEvent("sources", retrievedSources);

      const contextBlock = retrievedSources
        .map((src) => {
          const locationLabel = src.page
            ? `Page ${src.page}, Chunk ${src.chunkIndex}`
            : `Chunk ${src.chunkIndex}`;
          return `[${src.citationNumber}] (${locationLabel} · cosine similarity ${src.score.toFixed(3)})\n${src.text}`;
        })
        .join("\n\n---\n\n");

      const recentHistory = Array.isArray(history)
        ? history
            .slice(-6)
            .filter(
              (m) =>
                (m.role === "user" || m.role === "assistant") &&
                typeof m.content === "string" &&
                m.content.trim().length > 0
            )
            .map(
              (m) =>
                `${m.role === "user" ? "User" : "Assistant"}: ${m.content.trim()}`
            )
            .join("\n")
        : "";

      const systemInstruction = [
        `You are Chatter, a precise document Q&A assistant.`,
        `Your task is to answer questions strictly using ONLY the retrieved excerpts from the uploaded document "${doc.fileName}".`,
        `Rules:`,
        `1. Ground every factual claim directly in the provided excerpts and cite the supporting excerpt numbers inline like [1] or [1][2].`,
        `2. Never use outside knowledge, assumptions, or unstated facts. If the excerpts do not contain enough information to answer the question, explicitly say: "The uploaded document doesn't contain information to answer that." and mention what the closest excerpts cover if helpful.`,
        `3. Keep answers clear, direct, and well-structured using concise prose or bullet points when appropriate.`,
      ].join("\n");

      const prompt = [
        `RETRIEVED DOCUMENT EXCERPTS:`,
        contextBlock,
        recentHistory ? `\nRECENT CONVERSATION:\n${recentHistory}` : "",
        `\nCURRENT QUESTION:\n${question.trim()}`,
      ]
        .filter(Boolean)
        .join("\n\n");

      const stream = await withRetry(() =>
        ai.models.generateContentStream({
          model: "gemini-4-hydrogen",
          contents: prompt,
          config: {
            systemInstruction,
            thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
          },
        })
      );

      for await (const chunk of stream) {
        if (res.writableEnded) break;
        const deltaText = chunk.text;
        if (deltaText) {
          sendEvent("delta", { text: deltaText });
        }
      }

      sendEvent("done", {});
      res.end();
    } catch (err) {
      console.error("Chat streaming error:", err);
      sendEvent("error", { message: formatGeminiError(err) });
      res.end();
    }
  });

  if (process.env.NODE_ENV === "production") {
    const distPath = path.resolve(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*all", (_req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  } else {
    const { createServer: createViteServer } = await import("vite");
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  }

  const port = Number(process.env.PORT) || 3000;
  app.listen(port, "0.0.0.0", () => {
    console.log(`Chatter server listening on http://0.0.0.0:${port}`);
  });
}

createApp().catch((err) => {
  console.error("Failed to start server:", err);
  process.exit(1);
});
