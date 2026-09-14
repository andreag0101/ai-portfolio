import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import FilePicker from "../components/FilePicker";
import ResultImage from "../components/ResultImage";
import Stepper, { type Step } from "../components/Stepper";
import PointCloudViewer from "../components/PointCloudViewer";
import {
  getPrecomputed,
  runDisparity,
  getDisparitySample,
  dataUrlToFile,
  decodeBase64Float32,
  decodeBase64Uint8,
  type DisparityResult,
} from "../lib/api";

function buildSteps(result: DisparityResult): Step[] {
  const positions = decodeBase64Float32(result.pointCloud.positions);
  const colors = decodeBase64Uint8(result.pointCloud.colors);
  const sgbmPositions = decodeBase64Float32(result.sgbmPointCloud.positions);
  const sgbmColors = decodeBase64Uint8(result.sgbmPointCloud.colors);

  return [
    {
      title: "Stereo pair",
      description: "Same scene, camera shifted sideways. Rectified so matching points sit on the same row.",
      content: (
        <div className="grid grid-cols-2 gap-3">
          <ResultImage label="Left" src={result.left} />
          <ResultImage label="Right" src={result.right} />
        </div>
      ),
    },
    {
      title: "Dense matching",
      description: "For every pixel, a census transform (a \"brighter than center\" bit vector over its window) is compared against candidate windows in the other image via Hamming distance. Brighter = larger disparity = closer to the camera.",
      content: <ResultImage label="Raw disparity (grayscale)" src={result.disparityGray} />,
    },
    {
      title: "Colorized depth map",
      description: "Same values, false-colored for readability. Warm = close, cool = far.",
      content: <ResultImage label="Disparity map" src={result.disparity} />,
    },
    {
      title: "Where it goes wrong",
      description: `Matching each pixel's window independently has no way to prefer spatially-consistent answers, so it's unreliable in flat/textureless regions specifically: many windows there look nearly identical. A left-right consistency check (match left→right and right→left independently, keep only pixels where they agree) plus a raw match-confidence threshold rejects those unreliable pixels instead of guessing. Dark pixels below were rejected; kept ${(result.filteredKeepFrac * 100).toFixed(0)}% of the image.`,
      content: <ResultImage label="Filtered disparity (unreliable pixels marked dark)" src={result.filteredDisparity} />,
    },
    {
      title: "Semi-global matching",
      description: `OpenCV's Semi-Global Matching, for comparison: instead of scoring every pixel's window in isolation, it aggregates cost along several directions with a penalty for neighboring disparities that disagree, so a flat patch inherits a plausible answer from its surroundings instead of guessing blind. Kept ${(result.sgbmKeepFrac * 100).toFixed(0)}% of the image, and (on a Middlebury test with known ground truth) is noticeably more accurate than the from-scratch window matcher above.`,
      content: <ResultImage label="Semi-global matching disparity" src={result.sgbmDisparity} />,
    },
    {
      title: "3D reconstruction",
      description: "Each kept pixel's disparity converted to depth and unprojected into 3D, colored from the left photo. There's no real camera calibration for an arbitrary upload, so this uses a nominal focal length: the shape is relatively correct, not metrically accurate. Drag to rotate, scroll to zoom, right-drag to pan.",
      content: (
        <div className="space-y-6">
          <div>
            <p className="mb-1.5 text-xs font-medium text-neutral-500">
              From the from-scratch matcher, filtered ({result.pointCloud.numPoints.toLocaleString()} points)
            </p>
            <PointCloudViewer positions={positions} colors={colors} height={360} />
          </div>
          <div>
            <p className="mb-1.5 text-xs font-medium text-neutral-500">
              From semi-global matching ({result.sgbmPointCloud.numPoints.toLocaleString()} points)
            </p>
            <PointCloudViewer positions={sgbmPositions} colors={sgbmColors} height={360} />
          </div>
        </div>
      ),
    },
  ];
}

export default function Disparity() {
  const [leftFile, setLeftFile] = useState<File[]>([]);
  const [rightFile, setRightFile] = useState<File[]>([]);
  const [window_, setWindow] = useState(9);
  const [dMax, setDMax] = useState(50);
  const [loading, setLoading] = useState(false);
  const [sampleLoading, setSampleLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<DisparityResult | null>(null);
  const [isDefault, setIsDefault] = useState(false);

  useEffect(() => {
    getPrecomputed<DisparityResult>("disparity")
      .then((r) => {
        setResult(r);
        setIsDefault(true);
      })
      .catch(() => {});
  }, []);

  async function handleRun(left?: File, right?: File) {
    const l = left ?? leftFile[0];
    const r = right ?? rightFile[0];
    if (!l || !r) return;
    setLoading(true);
    setError(null);
    try {
      const res = await runDisparity(l, r, { window: window_, dMax });
      setResult(res);
      setIsDefault(false);
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
      const { left, right } = await getDisparitySample();
      const [leftF, rightF] = await Promise.all([
        dataUrlToFile(left, "middlebury-left.png"),
        dataUrlToFile(right, "middlebury-right.png"),
      ]);
      setLeftFile([leftF]);
      setRightFile([rightF]);
      await handleRun(leftF, rightF);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong");
    } finally {
      setSampleLoading(false);
    }
  }

  return (
    <div className="mx-auto max-w-5xl px-6 py-10">
      <Link to="/" className="text-sm text-neutral-500 hover:text-neutral-900 dark:hover:text-neutral-100">
        &larr; All demos
      </Link>
      <h1 className="mt-2 text-2xl font-semibold tracking-tight">Stereo Disparity</h1>
      <p className="mt-2 max-w-2xl text-sm text-neutral-600 dark:text-neutral-400">
        Upload a rectified stereo pair (same scene, camera shifted sideways; e.g. a Middlebury
        stereo set). For every pixel, a census transform encodes its window as a "brighter than
        center" bit vector; the disparity that minimizes the Hamming distance between left and right
        windows wins. Warm colors = close to the camera, cool colors = far away. On a Middlebury pair
        with known ground truth, this from-scratch matcher gets ~74&ndash;78% of pixels within 2px at
        typical window sizes: a real result, not a broken one, but far from perfect. The later
        steps show why, and two ways to fix it.
      </p>
      <p className="mt-2 max-w-2xl text-sm text-neutral-600 dark:text-neutral-400">
        This matters most for input quality: the algorithm only searches along rows, so it assumes
        the pair is <em>actually rectified</em>: matching points sit on exactly the same
        scanline. A casual pair of handheld phone photos almost never satisfies that (any hand tilt or
        vertical shift breaks the assumption entirely), so results from a phone-shot pair can look bad
        for reasons that have nothing to do with the matching algorithm itself. Use the sample button
        below for a known-good, properly rectified pair.
      </p>

      <div className="mt-8 grid grid-cols-1 gap-8 lg:grid-cols-[280px_1fr]">
        <div className="space-y-5">
          <FilePicker label="Left image" files={leftFile} onChange={setLeftFile} />
          <FilePicker label="Right image" files={rightFile} onChange={setRightFile} />

          <button
            onClick={handleUseSample}
            disabled={sampleLoading || loading}
            className="w-full rounded-md bg-neutral-100 px-4 py-2 text-sm font-medium text-neutral-700 transition disabled:opacity-40 dark:bg-neutral-800 dark:text-neutral-300"
          >
            {sampleLoading ? "Loading…" : "Use sample stereo pair"}
          </button>

          <div>
            <label className="mb-1.5 flex justify-between text-sm font-medium text-neutral-700 dark:text-neutral-300">
              <span>Window size</span>
              <span className="text-neutral-400">{window_}px</span>
            </label>
            <input
              type="range"
              min={3}
              max={19}
              step={2}
              value={window_}
              onChange={(e) => setWindow(Number(e.target.value))}
              className="w-full"
            />
          </div>

          <div>
            <label className="mb-1.5 flex justify-between text-sm font-medium text-neutral-700 dark:text-neutral-300">
              <span>Max disparity search</span>
              <span className="text-neutral-400">{dMax}px</span>
            </label>
            <input
              type="range"
              min={10}
              max={100}
              step={5}
              value={dMax}
              onChange={(e) => setDMax(Number(e.target.value))}
              className="w-full"
            />
          </div>

          <button
            onClick={() => handleRun()}
            disabled={leftFile.length === 0 || rightFile.length === 0 || loading}
            className="w-full rounded-md bg-neutral-900 px-4 py-2 text-sm font-medium text-white transition disabled:opacity-40 dark:bg-neutral-100 dark:text-neutral-900"
          >
            {loading ? "Matching…" : "Compute disparity"}
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
                  Showing the bundled sample result. Upload your own stereo pair to run it live.
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
