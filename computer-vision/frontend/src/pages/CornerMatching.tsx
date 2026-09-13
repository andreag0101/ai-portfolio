import { useState } from "react";
import { Link } from "react-router-dom";
import FilePicker from "../components/FilePicker";
import ResultImage from "../components/ResultImage";
import Stepper, { type Step } from "../components/Stepper";
import { runCornerMatch, getCornersSample, dataUrlToFile, type CornerMatchResult } from "../lib/api";

function buildSteps(result: CornerMatchResult, distType: "SSD" | "NCC"): Step[] {
  return [
    {
      title: "Interest points",
      description: "A from-scratch multiscale Harris detector: Haar-wavelet-style derivative filters, a second-moment matrix summed over a window, non-max suppression.",
      content: (
        <div className="grid grid-cols-2 gap-3">
          <ResultImage label={`Image 1 (${result.numCorners1} corners)`} src={result.cornersOverlay1} />
          <ResultImage label={`Image 2 (${result.numCorners2} corners)`} src={result.cornersOverlay2} />
        </div>
      ),
    },
    {
      title: "Matching",
      description: `A patch-based ${distType} comparison greedily pairs up the best mutual matches (${result.numMatches} found).`,
      content: <ResultImage label={`Harris corners + ${distType} matching (from scratch)`} src={result.combined} />,
    },
    {
      title: "ORB baseline",
      description: "OpenCV's ORB descriptor + brute-force Hamming matching, for comparison against the from-scratch approach.",
      content: <ResultImage label="ORB + brute-force Hamming matching (OpenCV baseline)" src={result.orbCombined} />,
    },
  ];
}

export default function CornerMatching() {
  const [files1, setFiles1] = useState<File[]>([]);
  const [files2, setFiles2] = useState<File[]>([]);
  const [sigma, setSigma] = useState(1.2);
  const [distType, setDistType] = useState<"SSD" | "NCC">("SSD");
  const [loading, setLoading] = useState(false);
  const [sampleLoading, setSampleLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<CornerMatchResult | null>(null);

  async function handleRun(file1?: File, file2?: File) {
    const f1 = file1 ?? files1[0];
    const f2 = file2 ?? files2[0];
    if (!f1 || !f2) return;
    setLoading(true);
    setError(null);
    try {
      setResult(await runCornerMatch(f1, f2, { sigma, distType }));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  async function handleUseSample() {
    setSampleLoading(true);
    setError(null);
    try {
      const { file1, file2 } = await getCornersSample();
      const [f1, f2] = await Promise.all([
        dataUrlToFile(file1, "hovde-1.jpg"),
        dataUrlToFile(file2, "hovde-2.jpg"),
      ]);
      setFiles1([f1]);
      setFiles2([f2]);
      await handleRun(f1, f2);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong");
    } finally {
      setSampleLoading(false);
    }
  }

  return (
    <div className="mx-auto max-w-5xl px-6 py-10">
      <Link to="/" className="text-sm text-neutral-500 hover:text-neutral-900 dark:hover:text-neutral-100">
        &larr; All projects
      </Link>
      <h1 className="mt-2 text-2xl font-semibold tracking-tight">Corner Detection & Feature Matching</h1>
      <p className="mt-2 max-w-2xl text-sm text-neutral-600 dark:text-neutral-400">
        A from-scratch multiscale Harris corner detector (Haar-wavelet-style derivative filters, a
        second-moment matrix, non-max suppression) finds interest points in each photo, then a
        patch-based SSD or NCC comparison greedily pairs up the best mutual matches. For contrast,
        OpenCV's ORB + brute-force Hamming matching runs alongside it as a modern baseline.
      </p>

      <div className="mt-8 grid grid-cols-1 gap-8 lg:grid-cols-[280px_1fr]">
        <div className="space-y-5">
          <p className="text-sm text-neutral-600 dark:text-neutral-400">
            Upload two photos of the same scene from slightly different viewpoints.
          </p>
          <FilePicker label="Image 1" files={files1} onChange={setFiles1} />
          <FilePicker label="Image 2" files={files2} onChange={setFiles2} />

          <button
            onClick={handleUseSample}
            disabled={sampleLoading || loading}
            className="w-full rounded-md bg-neutral-100 px-4 py-2 text-sm font-medium text-neutral-700 transition disabled:opacity-40 dark:bg-neutral-800 dark:text-neutral-300"
          >
            {sampleLoading ? "Loading…" : "Use sample photo pair"}
          </button>

          <div>
            <label className="mb-1.5 flex justify-between text-sm font-medium text-neutral-700 dark:text-neutral-300">
              <span>Corner scale (&sigma;)</span>
              <span className="text-neutral-400">{sigma.toFixed(1)}</span>
            </label>
            <input
              type="range"
              min={0.5}
              max={3}
              step={0.1}
              value={sigma}
              onChange={(e) => setSigma(Number(e.target.value))}
              className="w-full"
            />
          </div>

          <div>
            <p className="mb-1.5 text-sm font-medium text-neutral-700 dark:text-neutral-300">Patch distance</p>
            <div className="flex gap-2">
              {(["SSD", "NCC"] as const).map((d) => (
                <button
                  key={d}
                  onClick={() => setDistType(d)}
                  className={`rounded-md px-3 py-1.5 text-sm ${
                    distType === d
                      ? "bg-neutral-900 text-white dark:bg-neutral-100 dark:text-neutral-900"
                      : "bg-neutral-100 text-neutral-700 dark:bg-neutral-800 dark:text-neutral-300"
                  }`}
                >
                  {d}
                </button>
              ))}
            </div>
          </div>

          <button
            onClick={() => handleRun()}
            disabled={files1.length === 0 || files2.length === 0 || loading}
            className="w-full rounded-md bg-neutral-900 px-4 py-2 text-sm font-medium text-white transition disabled:opacity-40 dark:bg-neutral-100 dark:text-neutral-900"
          >
            {loading ? "Matching…" : "Detect & match"}
          </button>
          {error && <p className="text-sm text-red-600 dark:text-red-400">{error}</p>}
          {result && (
            <p className="text-sm text-neutral-500">
              {result.numCorners1} / {result.numCorners2} corners detected, {result.numMatches} matched
            </p>
          )}
        </div>

        <div className="space-y-4">
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
          {result && <Stepper steps={buildSteps(result, distType)} />}
        </div>
      </div>
    </div>
  );
}
