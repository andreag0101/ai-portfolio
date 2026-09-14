import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import FilePicker from "../components/FilePicker";
import ResultImage from "../components/ResultImage";
import Stepper, { type Step } from "../components/Stepper";
import { runSegmentation, getSegmentationSamples, getPrecomputed, dataUrlToFile, type SegmentationResult } from "../lib/api";

function buildSteps(result: SegmentationResult): Step[] {
  return [
    { title: "Input", content: <ResultImage label="Input (resized)" src={result.input} /> },
    {
      title: "Initial threshold",
      description: "A single pass of Otsu's method: pick the threshold that maximizes between-class variance.",
      content: <ResultImage label="Mask before iteration" src={result.maskBeforeIteration} />,
    },
    {
      title: "Refined mask",
      description: "Otsu re-run on just the foreground, iteratively, to tighten the boundary.",
      content: (
        <div className="grid grid-cols-2 gap-3">
          <ResultImage label="Mask after iteration" src={result.mask} />
          <ResultImage label="Contour (raw mask)" src={result.contour} />
        </div>
      ),
    },
    {
      title: "Opening",
      description: "Erosion then dilation: strips small speckle noise from the mask.",
      content: (
        <div className="grid grid-cols-2 gap-3">
          <ResultImage label="Opening mask" src={result.openingMask} />
          <ResultImage label="Opening contour" src={result.openingContour} />
        </div>
      ),
    },
    {
      title: "Closing",
      description: "Dilation then erosion: fills small holes inside the foreground region.",
      content: (
        <div className="grid grid-cols-2 gap-3">
          <ResultImage label="Closing mask" src={result.closingMask} />
          <ResultImage label="Closing contour" src={result.closingContour} />
        </div>
      ),
    },
  ];
}

export default function Segmentation() {
  const [files, setFiles] = useState<File[]>([]);
  const [mode, setMode] = useState<"rgb" | "texture">("rgb");
  const [bins, setBins] = useState(256);
  const [flip, setFlip] = useState(false);
  const [iterations, setIterations] = useState(1);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<SegmentationResult | null>(null);
  const [isDefault, setIsDefault] = useState(false);
  const [samples, setSamples] = useState<string[]>([]);

  useEffect(() => {
    getSegmentationSamples().then((r) => setSamples(r.samples)).catch(() => {});
    getPrecomputed<SegmentationResult>("segmentation")
      .then((r) => {
        setResult(r);
        setIsDefault(true);
      })
      .catch(() => {});
  }, []);

  async function handleRun(file?: File) {
    const f = file ?? files[0];
    if (!f) return;
    setLoading(true);
    setError(null);
    try {
      const res = await runSegmentation(f, { mode, bins, flip, iterations });
      setResult(res);
      setIsDefault(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  async function handleSampleClick(dataUrl: string, i: number) {
    const file = await dataUrlToFile(dataUrl, `sample-${i}.jpg`);
    setFiles([file]);
    setIsDefault(false);
    handleRun(file);
  }

  return (
    <div className="mx-auto max-w-5xl px-6 py-10">
      <Link to="/" className="text-sm text-neutral-500 hover:text-neutral-900 dark:hover:text-neutral-100">
        &larr; All demos
      </Link>
      <h1 className="mt-2 text-2xl font-semibold tracking-tight">Image Segmentation</h1>
      <p className="mt-2 max-w-2xl text-sm text-neutral-600 dark:text-neutral-400">
        From-scratch iterative Otsu thresholding &mdash; run per RGB channel or on a local-variance
        texture map &mdash; followed by morphological opening/closing and boundary extraction. Every
        mask is the raw, un-smoothed output of the algorithm, so noisy speckling in the initial mask
        is expected; opening/closing cleans it up.
      </p>

      <div className="mt-8 grid grid-cols-1 gap-8 lg:grid-cols-[280px_1fr]">
        <div className="space-y-5">
          <FilePicker label="Image" files={files} onChange={setFiles} />

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

          <div>
            <p className="mb-1.5 text-sm font-medium text-neutral-700 dark:text-neutral-300">Segmentation mode</p>
            <div className="flex gap-2">
              {(["rgb", "texture"] as const).map((m) => (
                <button
                  key={m}
                  onClick={() => setMode(m)}
                  className={`rounded-md px-3 py-1.5 text-sm capitalize ${
                    mode === m
                      ? "bg-neutral-900 text-white dark:bg-neutral-100 dark:text-neutral-900"
                      : "bg-neutral-100 text-neutral-700 dark:bg-neutral-800 dark:text-neutral-300"
                  }`}
                >
                  {m}
                </button>
              ))}
            </div>
          </div>

          {mode === "rgb" && (
            <div>
              <label className="mb-1.5 flex justify-between text-sm font-medium text-neutral-700 dark:text-neutral-300">
                <span>Histogram bins</span>
                <span className="text-neutral-400">{bins}</span>
              </label>
              <input
                type="range"
                min={16}
                max={256}
                step={8}
                value={bins}
                onChange={(e) => setBins(Number(e.target.value))}
                className="w-full"
              />
            </div>
          )}

          <div>
            <label className="mb-1.5 flex justify-between text-sm font-medium text-neutral-700 dark:text-neutral-300">
              <span>Refinement iterations</span>
              <span className="text-neutral-400">{iterations}</span>
            </label>
            <input
              type="range"
              min={0}
              max={5}
              value={iterations}
              onChange={(e) => setIterations(Number(e.target.value))}
              className="w-full"
            />
          </div>

          <label className="flex items-center gap-2 text-sm text-neutral-700 dark:text-neutral-300">
            <input type="checkbox" checked={flip} onChange={(e) => setFlip(e.target.checked)} />
            Flip foreground/background
          </label>

          <button
            onClick={() => handleRun()}
            disabled={files.length === 0 || loading}
            className="w-full rounded-md bg-neutral-900 px-4 py-2 text-sm font-medium text-white transition disabled:opacity-40 dark:bg-neutral-100 dark:text-neutral-900"
          >
            {loading ? "Segmenting…" : "Run segmentation"}
          </button>

          {error && <p className="text-sm text-red-600 dark:text-red-400">{error}</p>}
        </div>

        <div>
          {!result && !loading && (
            <div className="flex h-64 items-center justify-center rounded-lg border border-dashed border-neutral-300 text-sm text-neutral-400 dark:border-neutral-700">
              Results will appear here
            </div>
          )}
          {loading && (
            <div className="flex h-64 items-center justify-center rounded-lg border border-dashed border-neutral-300 text-sm text-neutral-400 dark:border-neutral-700">
              Processing…
            </div>
          )}
          {result && (
            <>
              {isDefault && (
                <p className="mb-3 text-sm text-neutral-500">
                  Showing the bundled sample result. Upload your own photo to run it live.
                </p>
              )}
              <Stepper steps={buildSteps(result)} />
            </>
          )}
        </div>
      </div>
    </div>
  );
}
