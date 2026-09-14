import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import FilePicker from "../components/FilePicker";
import ResultImage from "../components/ResultImage";
import Stepper, { type Step } from "../components/Stepper";
import {
  runCalibration,
  runCalibrationSample,
  getCalibrationPattern,
  getCalibrationSamples,
  getPrecomputed,
  type CalibrationResult,
} from "../lib/api";

function buildSteps(result: CalibrationResult): Step[] {
  return [
    {
      title: "Corner detection",
      description: "Canny edges → Hough line detection → every vertical/horizontal line intersection is a corner candidate → k-means clusters those down to the 80 true corners (20 squares × 4 corners).",
      content: (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
          {result.cornerOverlays.map((src, i) => (
            <ResultImage key={i} label={`Photo ${i + 1}`} src={src} />
          ))}
        </div>
      ),
    },
    {
      title: "Sorting into a grid",
      description: "The unordered 80 corners are sorted into an 8-wide × 10-tall grid (via a homography to a canonical rectangle), so the same physical point has the same index across every photo.",
      content: (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
          {result.labeledOverlays.map((src, i) => (
            <ResultImage key={i} label={`Photo ${i + 1}`} src={src} />
          ))}
        </div>
      ),
    },
    {
      title: "Recovered intrinsics",
      description: "Each photo gives a homography (model plane → image); combining all of them via Zhang's absolute-conic method gives a single closed-form estimate of the camera's intrinsic matrix, shared across every photo.",
      content: (
        <div className="grid max-w-sm grid-cols-2 gap-x-6 gap-y-2 text-sm">
          <span className="text-neutral-500">Focal length (fx, fy)</span>
          <span className="font-mono">{result.intrinsics.fx.toFixed(1)}, {result.intrinsics.fy.toFixed(1)}</span>
          <span className="text-neutral-500">Principal point (cx, cy)</span>
          <span className="font-mono">{result.intrinsics.cx.toFixed(1)}, {result.intrinsics.cy.toFixed(1)}</span>
          <span className="text-neutral-500">Skew</span>
          <span className="font-mono">{result.intrinsics.skew.toFixed(4)}</span>
        </div>
      ),
    },
    {
      title: "Reprojection accuracy",
      description: "Green = detected corners. Red = the model grid reprojected using the recovered pose, before vs. after Levenberg-Marquardt refinement. Tighter overlap after LM means a better fit.",
      content: (
        <div className="space-y-4">
          {result.reprojBefore.map((src, i) => (
            <div key={i} className="grid grid-cols-2 gap-3">
              <ResultImage label={`Photo ${i + 1}: before LM`} src={src} />
              <ResultImage label={`Photo ${i + 1}: after LM`} src={result.reprojAfter[i]} />
            </div>
          ))}
        </div>
      ),
    },
    {
      title: "3D camera poses",
      description: "Each photo's recovered position and orientation relative to the calibration plane, plotted in 3D.",
      content: <ResultImage label="Recovered camera poses" src={result.posePlot} />,
    },
  ];
}

export default function Calibration() {
  const [files, setFiles] = useState<File[]>([]);
  const [pattern, setPattern] = useState<string | null>(null);
  const [samples, setSamples] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [statusMsg, setStatusMsg] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<CalibrationResult | null>(null);
  const [isDefault, setIsDefault] = useState(false);

  useEffect(() => {
    getCalibrationPattern().then((r) => setPattern(r.pattern)).catch(() => {});
    getCalibrationSamples().then((r) => setSamples(r.samples)).catch(() => {});
    getPrecomputed<CalibrationResult>("calibration")
      .then((r) => {
        setResult(r);
        setIsDefault(true);
      })
      .catch(() => {});
  }, []);

  async function handleRun() {
    if (files.length < 3) return;
    setLoading(true);
    setError(null);
    setStatusMsg("Detecting corners and solving for intrinsics — a few seconds…");
    try {
      setResult(await runCalibration(files));
      setIsDefault(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong");
    } finally {
      setLoading(false);
      setStatusMsg(null);
    }
  }

  async function handleRunSample() {
    setLoading(true);
    setError(null);
    setFiles([]);
    setStatusMsg("Running on the bundled sample photos…");
    try {
      setResult(await runCalibrationSample());
      setIsDefault(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong");
    } finally {
      setLoading(false);
      setStatusMsg(null);
    }
  }

  return (
    <div className="mx-auto max-w-5xl px-6 py-10">
      <Link to="/" className="text-sm text-neutral-500 hover:text-neutral-900 dark:hover:text-neutral-100">
        &larr; All projects
      </Link>
      <h1 className="mt-2 text-2xl font-semibold tracking-tight">Camera Calibration (Zhang's Method)</h1>
      <p className="mt-2 max-w-2xl text-sm text-neutral-600 dark:text-neutral-400">
        Recovers a camera's intrinsic parameters (focal length, principal point) from 3+ photos of a
        planar pattern at different angles &mdash; a from-scratch Hough-line corner detector, per-photo
        homographies, Zhang's closed-form absolute-conic solution, and Levenberg-Marquardt refinement.
        The pattern is a specific printable grid of squares (not a standard chessboard), so calibrating
        your own photos means printing it below; otherwise try the bundled sample photos.
      </p>

      <div className="mt-8 grid grid-cols-1 gap-8 lg:grid-cols-[300px_1fr]">
        <div className="space-y-5">
          {pattern && (
            <div>
              <p className="mb-1.5 text-sm font-medium text-neutral-700 dark:text-neutral-300">Calibration pattern</p>
              <img src={pattern} alt="Calibration pattern" className="w-full rounded-md border border-neutral-200 dark:border-neutral-800" />
              <p className="mt-1 text-xs text-neutral-500">Print this, then photograph it from 3+ angles.</p>
            </div>
          )}

          <FilePicker label="Photos of the pattern (3+)" files={files} onChange={setFiles} multiple />
          <button
            onClick={handleRun}
            disabled={files.length < 3 || loading}
            className="w-full rounded-md bg-neutral-900 px-4 py-2 text-sm font-medium text-white transition disabled:opacity-40 dark:bg-neutral-100 dark:text-neutral-900"
          >
            {loading ? "Calibrating…" : "Calibrate"}
          </button>

          <div className="relative">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-neutral-200 dark:border-neutral-800" />
            </div>
            <div className="relative flex justify-center text-xs">
              <span className="bg-neutral-50 px-2 text-neutral-400 dark:bg-neutral-950">or</span>
            </div>
          </div>

          <button
            onClick={handleRunSample}
            disabled={loading}
            className="w-full rounded-md bg-neutral-100 px-4 py-2 text-sm font-medium text-neutral-700 transition disabled:opacity-40 dark:bg-neutral-800 dark:text-neutral-300"
          >
            Use sample photos
          </button>
          {samples.length > 0 && (
            <div className="flex flex-wrap gap-1.5">
              {samples.map((s, i) => (
                <img key={i} src={s} alt={`Sample ${i + 1}`} className="h-10 w-14 rounded border border-neutral-200 object-cover dark:border-neutral-800" />
              ))}
            </div>
          )}

          {statusMsg && <p className="text-sm text-neutral-500">{statusMsg}</p>}
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
                  Showing the bundled sample result. Upload your own photos to run it live.
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
