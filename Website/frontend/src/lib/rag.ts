const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "/api";

async function unwrap<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (body?.detail) detail = body.detail;
    } catch {
      // ignore non-JSON error bodies
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export interface RagCitation {
  title: string;
  page: number;
  snippet: string;
  pdfHref: string;
  projectHref: string;
  score?: number;
}

export interface RagAnswer {
  answer: string;
  mode: "generated" | "extractive" | "none";
  citations: RagCitation[];
}

export interface RagSuggestedItem extends RagAnswer {
  question: string;
}

export interface RagDocument {
  file: string;
  title: string;
  pdfHref: string;
  projectHref: string;
}

export async function getRagDocuments(): Promise<{ documents: RagDocument[] }> {
  const res = await fetch(`${API_BASE}/rag/documents`);
  return unwrap(res);
}

export async function getRagSuggested(): Promise<{ suggested: RagSuggestedItem[] }> {
  const res = await fetch(`${API_BASE}/rag/suggested`);
  return unwrap(res);
}

export async function askRag(question: string): Promise<RagAnswer> {
  const res = await fetch(`${API_BASE}/rag/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  return unwrap(res);
}
