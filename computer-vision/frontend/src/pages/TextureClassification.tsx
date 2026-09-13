import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import FilePicker from "../components/FilePicker";
import ResultImage from "../components/ResultImage";
import Stepper, { type Step } from "../components/Stepper";
import { runTextureClassify, getTextureSamples, dataUrlToFile, type TextureResult } from "../lib/api";

const CLASS_LABELS: Record<string, string> = {
  cloudy: "Cloudy",
  rain: "Rain",
  shine: "Shine",
  sunrise: "Sunrise",
};

function PredictionPanel({ result }: { result: TextureResult }) {
  return (
    <>
      <div className="rounded-lg border border-emerald-300 bg-emerald-50 p-4 dark:border-emerald-800 dark:bg-emerald-950/30">
        <p className="text-xs font-medium uppercase tracking-wide text-neutral-500">Predicted</p>
        <p className="mt-1 text-lg font-semibold text-emerald-700 dark:text-emerald-400">
          {CLASS_LABELS[result.predicted] ?? result.predicted}
        </p>
      </div>

      <div className="mt-4 space-y-2">
        <p className="text-sm font-medium text-neutral-700 dark:text-neutral-300">Nearest-neighbor votes</p>
        {Object.entries(result.confidences).map(([cls, conf]) => (
          <div key={cls} className="flex items-center gap-3 text-sm">
            <span className="w-16 text-neutral-600 dark:text-neutral-400">{CLASS_LABELS[cls] ?? cls}</span>
            <div className="h-2 flex-1 overflow-hidden rounded-full bg-neutral-100 dark:bg-neutral-800">
              <div className="h-full bg-emerald-500" style={{ width: `${conf * 100}%` }} />
            </div>
            <span className="w-10 text-right text-neutral-400">{Math.round(conf * 100)}%</span>
          </div>
        ))}
      </div>

      <div className="mt-4">
        <p className="mb-2 text-sm font-medium text-neutral-700 dark:text-neutral-300">Closest training photos</p>
        <div className="flex flex-wrap gap-2">
          {result.nearest.map((n, i) => (
            <figure key={i} className="overflow-hidden rounded-md border border-neutral-200 dark:border-neutral-800">
              <img src={n.thumbnail} alt={n.label} className="h-20 w-20 object-cover" />
              <figcaption className="bg-neutral-50 px-1.5 py-0.5 text-center text-xs text-neutral-500 dark:bg-neutral-900">
                {CLASS_LABELS[n.label] ?? n.label}
              </figcaption>
            </figure>
          ))}
        </div>
      </div>
    </>
  );
}

function buildSteps(result: TextureResult): Step[] {
  return [
    {
      title: "Input",
      description: "Resized to 64×64 to match the training set.",
      content: <ResultImage label="Input" src={result.resizedInput} />,
    },
    {
      title: "Hue channel",
      description: "The feature is computed on Hue alone, not the full RGB image.",
      content: <ResultImage label="Hue" src={result.hueVisual} />,
    },
    {
      title: "LBP texture map",
      description: "Each pixel's rotation-invariant local binary pattern, color-coded by which of the 10 pattern categories it falls into. This per-pixel map is summarized into a 10-bin histogram — that histogram is the whole feature vector.",
      content: <ResultImage label="LBP encoding" src={result.encodingVisual} />,
    },
    {
      title: "Prediction",
      description: "1-nearest-neighbor over 922 training histograms (histogram-intersection distance).",
      content: <PredictionPanel result={result} />,
    },
  ];
}

export default function TextureClassification() {
  const [files, setFiles] = useState<File[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<TextureResult | null>(null);
  const [samples, setSamples] = useState<string[]>([]);

  useEffect(() => {
    getTextureSamples().then((r) => setSamples(r.samples)).catch(() => {});
  }, []);

  async function handleRun(file?: File) {
    const f = file ?? files[0];
    if (!f) return;
    setLoading(true);
    setError(null);
    try {
      setResult(await runTextureClassify(f));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  async function handleSampleClick(dataUrl: string, i: number) {
    const file = await dataUrlToFile(dataUrl, `sample-${i}.jpg`);
    setFiles([file]);
    handleRun(file);
  }

  return (
    <div className="mx-auto max-w-5xl px-6 py-10">
      <Link to="/" className="text-sm text-neutral-500 hover:text-neutral-900 dark:hover:text-neutral-100">
        &larr; All projects
      </Link>
      <h1 className="mt-2 text-2xl font-semibold tracking-tight">Texture-Based Weather Classification</h1>
      <p className="mt-2 max-w-2xl text-sm text-neutral-600 dark:text-neutral-400">
        A rotation-invariant Local Binary Pattern histogram (computed on the Hue channel) describes each
        photo's texture, then 1-nearest-neighbor over a bundled set of 922 labeled photos (cloudy, rain,
        shine, sunrise) picks the class. It's a deliberately simple, interpretable feature &mdash; on a
        held-out set of 200 photos it reaches about <strong>57% accuracy</strong> across 4 classes (vs.
        25% chance), clearly better than guessing but not a strong classifier. Expect it to lean on sky
        color and texture more than genuine weather cues.
      </p>

      <div className="mt-8 grid grid-cols-1 gap-8 lg:grid-cols-[280px_1fr]">
        <div className="space-y-5">
          <FilePicker label="Photo" files={files} onChange={setFiles} />

          {samples.length > 0 && (
            <div>
              <p className="mb-1.5 text-sm font-medium text-neutral-700 dark:text-neutral-300">Or try a sample</p>
              <div className="flex flex-wrap gap-1.5">
                {samples.map((s, i) => (
                  <button key={i} onClick={() => handleSampleClick(s, i)} className="overflow-hidden rounded-md border border-neutral-200 dark:border-neutral-800">
                    <img src={s} alt={`Sample ${i + 1}`} className="h-14 w-14 object-cover" />
                  </button>
                ))}
              </div>
            </div>
          )}

          <button
            onClick={() => handleRun()}
            disabled={files.length === 0 || loading}
            className="w-full rounded-md bg-neutral-900 px-4 py-2 text-sm font-medium text-white transition disabled:opacity-40 dark:bg-neutral-100 dark:text-neutral-900"
          >
            {loading ? "Classifying…" : "Classify"}
          </button>
          {error && <p className="text-sm text-red-600 dark:text-red-400">{error}</p>}
        </div>

        <div className="space-y-6">
          {!result && !loading && (
            <div className="flex h-64 items-center justify-center rounded-lg border border-dashed border-neutral-300 text-sm text-neutral-400 dark:border-neutral-700">
              Results will appear here
            </div>
          )}
          {loading && (
            <div className="flex h-64 items-center justify-center rounded-lg border border-dashed border-neutral-300 text-sm text-neutral-400 dark:border-neutral-700">
              Comparing against 922 training photos…
            </div>
          )}
          {result && <Stepper steps={buildSteps(result)} />}
        </div>
      </div>
    </div>
  );
}
