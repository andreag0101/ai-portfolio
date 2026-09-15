import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { askRag, getRagSuggested, type RagAnswer, type RagCitation } from "../lib/rag";

interface Turn {
  question: string;
  answer: RagAnswer;
}

function ChatIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" className="h-6 w-6">
      <path
        d="M4 4.5h16a1 1 0 0 1 1 1V15a1 1 0 0 1-1 1H9l-4.5 4V16H4a1 1 0 0 1-1-1V5.5a1 1 0 0 1 1-1Z"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function CloseIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" className="h-4 w-4">
      <path d="M6 6l12 12M18 6L6 18" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
    </svg>
  );
}

function ModeBadge({ mode }: { mode: RagAnswer["mode"] }) {
  if (mode === "generated") {
    return (
      <span className="inline-flex items-center rounded-full bg-gold-50 px-2 py-0.5 text-[10px] font-medium text-gold-700 ring-1 ring-inset ring-gold-200 dark:bg-gold-950/30 dark:text-gold-400 dark:ring-gold-800">
        Claude, grounded in citations
      </span>
    );
  }
  if (mode === "extractive") {
    return (
      <span className="inline-flex items-center rounded-full bg-neutral-100 px-2 py-0.5 text-[10px] font-medium text-neutral-600 ring-1 ring-inset ring-neutral-200 dark:bg-neutral-800 dark:text-neutral-400 dark:ring-neutral-700">
        Retrieved passage
      </span>
    );
  }
  return (
    <span className="inline-flex items-center rounded-full bg-neutral-100 px-2 py-0.5 text-[10px] font-medium text-neutral-600 ring-1 ring-inset ring-neutral-200 dark:bg-neutral-800 dark:text-neutral-400 dark:ring-neutral-700">
      No match found
    </span>
  );
}

function CitationRow({ citation }: { citation: RagCitation }) {
  return (
    <div className="rounded-md border border-neutral-200 bg-neutral-50 p-2 text-xs dark:border-neutral-800 dark:bg-neutral-900/60">
      <div className="flex items-baseline justify-between gap-2">
        <p className="font-medium text-neutral-700 dark:text-neutral-300">{citation.title}</p>
        <span className="shrink-0 text-neutral-400">p.{citation.page}</span>
      </div>
      <p className="mt-0.5 line-clamp-2 text-neutral-500 dark:text-neutral-500">&ldquo;{citation.snippet}&rdquo;</p>
      <div className="mt-1 flex flex-wrap gap-3">
        <a href={citation.pdfHref} target="_blank" rel="noreferrer" className="font-medium text-gold-700 hover:underline dark:text-gold-400">
          Open PDF &rarr;
        </a>
        <Link to={citation.projectHref} className="font-medium text-gold-700 hover:underline dark:text-gold-400">
          View write-up &rarr;
        </Link>
      </div>
    </div>
  );
}

function TurnBubble({ turn }: { turn: Turn }) {
  return (
    <div className="space-y-2">
      <div className="ml-auto max-w-[85%] rounded-2xl rounded-br-sm bg-neutral-900 px-3 py-2 text-sm text-white dark:bg-neutral-100 dark:text-neutral-900">
        {turn.question}
      </div>
      <div className="mr-auto max-w-[92%] space-y-2 rounded-2xl rounded-bl-sm border border-neutral-200 bg-white px-3 py-2.5 dark:border-neutral-800 dark:bg-neutral-900">
        <ModeBadge mode={turn.answer.mode} />
        <p className="text-sm leading-relaxed text-neutral-700 dark:text-neutral-300">{turn.answer.answer}</p>
        {turn.answer.citations.length > 0 && (
          <div className="space-y-1.5 pt-0.5">
            {turn.answer.citations.map((c, i) => (
              <CitationRow key={i} citation={c} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default function RagChatWidget() {
  const [open, setOpen] = useState(false);
  const [suggested, setSuggested] = useState<(RagAnswer & { question: string })[]>([]);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (open && suggested.length === 0) {
      getRagSuggested()
        .then((r) => setSuggested(r.suggested))
        .catch(() => {});
    }
  }, [open, suggested.length]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [turns, loading]);

  function askSuggested(item: RagAnswer & { question: string }) {
    setTurns((t) => [...t, { question: item.question, answer: item }]);
  }

  async function askLive(q: string) {
    const trimmed = q.trim();
    if (!trimmed) return;
    setLoading(true);
    setError(null);
    setQuestion("");
    try {
      const answer = await askRag(trimmed);
      setTurns((t) => [...t, { question: trimmed, answer }]);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="fixed bottom-5 right-5 z-50 flex flex-col items-end gap-3 sm:bottom-6 sm:right-6">
      {open && (
        <div className="flex h-[min(600px,calc(100vh-7rem))] w-[min(380px,calc(100vw-2.5rem))] flex-col overflow-hidden rounded-2xl border border-neutral-200 bg-white shadow-2xl dark:border-neutral-800 dark:bg-neutral-950">
          <div className="flex shrink-0 items-start justify-between gap-2 border-b border-neutral-200 bg-neutral-50 px-4 py-3 dark:border-neutral-800 dark:bg-neutral-900">
            <div>
              <p className="text-sm font-semibold text-neutral-900 dark:text-neutral-100">Research Assistant</p>
              <p className="text-xs text-neutral-500 dark:text-neutral-400">RAG over my deep learning write-ups</p>
            </div>
            <button
              onClick={() => setOpen(false)}
              aria-label="Close chat"
              className="shrink-0 rounded-md p-1 text-neutral-400 transition hover:bg-neutral-200 hover:text-neutral-700 dark:hover:bg-neutral-800 dark:hover:text-neutral-200"
            >
              <CloseIcon />
            </button>
          </div>

          <div ref={scrollRef} className="flex-1 space-y-4 overflow-y-auto px-3 py-3">
            {turns.length === 0 && (
              <div className="space-y-3">
                <div className="rounded-xl border border-dashed border-neutral-300 p-3 text-xs leading-relaxed text-neutral-500 dark:border-neutral-700 dark:text-neutral-400">
                  Ask about backprop, GANs, object detection, transformers&hellip; Retrieval is a from-scratch
                  TF-IDF index over 10 write-ups; Claude generates the cited answer.
                </div>
                {suggested.length > 0 && (
                  <div>
                    <p className="mb-1.5 text-xs font-medium uppercase tracking-wide text-neutral-400">
                      Try asking
                    </p>
                    <div className="space-y-1.5">
                      {suggested.map((item) => (
                        <button
                          key={item.question}
                          onClick={() => askSuggested(item)}
                          className="block w-full rounded-md border border-neutral-200 px-2.5 py-1.5 text-left text-xs text-neutral-600 transition hover:border-gold-300 hover:text-gold-700 dark:border-neutral-800 dark:text-neutral-400 dark:hover:border-gold-800 dark:hover:text-gold-400"
                        >
                          {item.question}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {turns.map((t, i) => (
              <TurnBubble key={i} turn={t} />
            ))}

            {loading && (
              <div className="mr-auto flex max-w-[85%] items-center gap-1.5 rounded-2xl rounded-bl-sm border border-neutral-200 bg-white px-3 py-2.5 dark:border-neutral-800 dark:bg-neutral-900">
                <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-neutral-400 [animation-delay:-0.3s]" />
                <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-neutral-400 [animation-delay:-0.15s]" />
                <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-neutral-400" />
              </div>
            )}
            {error && <p className="text-xs text-red-600 dark:text-red-400">{error}</p>}
          </div>

          <div className="shrink-0 border-t border-neutral-200 p-2.5 dark:border-neutral-800">
            <form
              onSubmit={(e) => {
                e.preventDefault();
                askLive(question);
              }}
              className="flex gap-1.5"
            >
              <input
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Ask a question…"
                maxLength={300}
                className="flex-1 rounded-md border border-neutral-300 bg-white px-3 py-2 text-sm text-neutral-900 placeholder:text-neutral-400 focus:border-gold-400 focus:outline-none dark:border-neutral-700 dark:bg-neutral-900 dark:text-neutral-100"
              />
              <button
                type="submit"
                disabled={loading || !question.trim()}
                className="shrink-0 rounded-md bg-neutral-900 px-3 py-2 text-sm font-medium text-white transition hover:bg-neutral-700 disabled:opacity-50 dark:bg-neutral-100 dark:text-neutral-900 dark:hover:bg-neutral-300"
              >
                Ask
              </button>
            </form>
            <Link
              to="/research-assistant"
              onClick={() => setOpen(false)}
              className="mt-2 block text-center text-[11px] font-medium text-neutral-400 hover:text-gold-700 dark:hover:text-gold-400"
            >
              Open full page &rarr;
            </Link>
          </div>
        </div>
      )}

      <button
        onClick={() => setOpen((o) => !o)}
        aria-label={open ? "Close chat" : "Open research assistant chat"}
        className="flex h-14 w-14 items-center justify-center rounded-full bg-neutral-900 text-white shadow-lg ring-1 ring-black/5 transition hover:bg-neutral-700 dark:bg-gold-600 dark:hover:bg-gold-500"
      >
        {open ? <CloseIcon /> : <ChatIcon />}
      </button>
    </div>
  );
}
