import { useEffect, useState } from "react";
import { findQuadCandidates, type Quad, type QuadCandidate } from "../lib/api";

interface QuadPickerProps {
  file: File | null;
  onQuadChange: (quad: Quad | null) => void;
}

/**
 * Runs automatic quadrilateral detection on `file` and shows the top few
 * candidates as clickable thumbnails. Detection is inherently ambiguous when
 * a photo has nested rectangles (e.g. a picture frame's outer edge vs. its
 * inner mat vs. the photo inside it) or partial occlusion, so rather than
 * silently committing to one guess, this surfaces the top candidates and lets
 * the user correct a wrong pick with one click instead of none at all.
 */
export default function QuadPicker({ file, onQuadChange }: QuadPickerProps) {
  const [candidates, setCandidates] = useState<QuadCandidate[]>([]);
  const [selected, setSelected] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!file) {
      setCandidates([]);
      setError(null);
      onQuadChange(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);
    findQuadCandidates(file)
      .then((res) => {
        if (cancelled) return;
        setCandidates(res.candidates);
        setSelected(0);
        onQuadChange(res.candidates[0]?.quad ?? null);
      })
      .catch((e) => {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : "Couldn't detect a region in this photo");
        setCandidates([]);
        onQuadChange(null);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [file]);

  if (!file) return null;

  if (loading) {
    return <p className="text-sm text-neutral-500">Detecting regions…</p>;
  }

  if (error) {
    return <p className="text-sm text-red-600 dark:text-red-400">{error}</p>;
  }

  if (candidates.length === 0) return null;

  return (
    <div>
      <p className="mb-1.5 text-sm font-medium text-neutral-700 dark:text-neutral-300">
        Detected region {candidates.length > 1 && <span className="text-neutral-400">(pick one if the first guess is wrong)</span>}
      </p>
      <div className="flex flex-wrap gap-2">
        {candidates.map((c, i) => (
          <button
            key={i}
            onClick={() => {
              setSelected(i);
              onQuadChange(c.quad);
            }}
            className={`overflow-hidden rounded-md border-2 transition ${
              selected === i ? "border-emerald-500" : "border-transparent hover:border-neutral-300"
            }`}
          >
            <img src={c.overlay} alt={`Option ${i + 1}`} className="h-20 w-20 object-cover" />
          </button>
        ))}
      </div>
    </div>
  );
}
