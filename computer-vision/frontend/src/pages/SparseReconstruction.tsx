import { useState } from "react";
import { Link } from "react-router-dom";
import FilePicker from "../components/FilePicker";
import ResultImage from "../components/ResultImage";
import Stepper, { type Step } from "../components/Stepper";
import PointCloudViewer from "../components/PointCloudViewer";
import {
  runSparseReconstruction,
  getSparseReconstructionSample,
  dataUrlToFile,
  decodeBase64Float32,
  decodeBase64Uint8,
  type SparseReconstructionResult,
} from "../lib/api";

function buildSteps(result: SparseReconstructionResult): Step[] {
  const positions = decodeBase64Float32(result.pointCloud.positions);
  const colors = decodeBase64Uint8(result.pointCloud.colors);

  return [
    {
      title: "Input pair",
      description: "Two photos of the same scene from different viewpoints — no rectification or calibration required going in.",
      content: (
        <div className="grid grid-cols-2 gap-3">
          <ResultImage label="Left" src={result.left} />
          <ResultImage label="Right" src={result.right} />
        </div>
      ),
    },
    {
      title: "Keypoint matching & epipolar geometry",
      description: `SIFT finds ${result.numSiftMatches} candidate correspondences; RANSAC (scored by Sampson error against the normalized 8-point fundamental matrix) keeps the ${result.numRansacInliers} that agree on a single consistent two-view geometry.`,
      content: <ResultImage label="Inlier correspondences" src={result.keypointMatchOverlay} />,
    },
    {
      title: "Rectification",
      description: "The recovered epipolar geometry gives a pair of homographies (one per image) that warp both views so corresponding points fall on the same image row — turning the 2D search for edge correspondences into a 1D one.",
      content: (
        <div className="grid grid-cols-2 gap-3">
          <ResultImage label="Rectified left" src={result.rectifiedLeft} />
          <ResultImage label="Rectified right" src={result.rectifiedRight} />
        </div>
      ),
    },
    {
      title: "Edge detection",
      description: "Canny edges on each rectified image. Only these interest points get matched and reconstructed — flat, textureless regions of the scene are skipped entirely, which is what keeps the final point cloud from being swamped by noisy background.",
      content: (
        <div className="grid grid-cols-2 gap-3">
          <ResultImage label="Edges, left" src={result.edgesLeft} />
          <ResultImage label="Edges, right" src={result.edgesRight} />
        </div>
      ),
    },
    {
      title: "Edge matching",
      description: `Each edge pixel is matched to an edge pixel on the same row of the other rectified image by windowed SSD, giving ${result.numEdgeMatches} correspondences.`,
      content: <ResultImage label="Edge correspondences (sampled)" src={result.edgeMatchOverlay} />,
    },
    {
      title: "3D reconstruction",
      description: `Matched edge points are mapped back to the original (unrectified) images and triangulated with a calibrated camera pair (essential matrix decomposition from a nominal focal length, with a cheirality check to pick the correct pose), giving ${result.numReconstructed.toLocaleString()} reconstructed points, colored from the left photo. Median parallax between the two viewing rays: ${result.medianParallaxDeg.toFixed(1)}° — above the ~1.5° floor this demo requires, so triangulation is well-conditioned rather than noise-dominated. Because only edge points are triangulated, there's no noisy background cloud to see through. Points are rendered as soft splats sized to overlap their neighbors (classical point-splatting, not the 2023 per-scene-trained "3D Gaussian Splatting" technique, which needs dozens of calibrated photos) so a sparse cloud reads as a filled surface instead of scattered dots. Drag to rotate, scroll to zoom, right-drag to pan.`,
      content: (
        <div>
          <p className="mb-1.5 text-xs font-medium text-neutral-500">
            {result.pointCloud.numPoints.toLocaleString()} points
          </p>
          <PointCloudViewer positions={positions} colors={colors} height={440} />
        </div>
      ),
    },
  ];
}

export default function SparseReconstruction() {
  const [leftFile, setLeftFile] = useState<File[]>([]);
  const [rightFile, setRightFile] = useState<File[]>([]);
  const [loading, setLoading] = useState(false);
  const [sampleLoading, setSampleLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<SparseReconstructionResult | null>(null);

  async function handleRun(left?: File, right?: File) {
    const l = left ?? leftFile[0];
    const r = right ?? rightFile[0];
    if (!l || !r) return;
    setLoading(true);
    setError(null);
    try {
      const res = await runSparseReconstruction(l, r);
      setResult(res);
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
      const { left, right } = await getSparseReconstructionSample();
      const [leftF, rightF] = await Promise.all([
        dataUrlToFile(left, "temple-left.jpg"),
        dataUrlToFile(right, "temple-right.jpg"),
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
        &larr; All projects
      </Link>
      <h1 className="mt-2 text-2xl font-semibold tracking-tight">Sparse Stereo Reconstruction (Edge-Based)</h1>
      <p className="mt-2 max-w-2xl text-sm text-neutral-600 dark:text-neutral-400">
        Unlike the dense disparity demo, this only triangulates edge/interest points instead of every
        pixel, so a cluttered or textured background never overwhelms the reconstruction of the actual
        subject. It doesn't need a pre-rectified pair either &mdash; it recovers the epipolar geometry
        from SIFT correspondences via RANSAC, rectifies the images itself, then matches and reconstructs
        only points that survive Canny edge detection.
      </p>
      <p className="mt-2 max-w-2xl text-sm text-neutral-600 dark:text-neutral-400">
        This needs enough reliable correspondences and, critically, enough <em>sideways</em> camera
        translation between the two shots (not just a tilt or pan in place). With too little of that
        translation, the two viewing rays to a point are nearly parallel, and even a pixel-perfect match
        swings wildly in depth once triangulated &mdash; noise that no amount of point filtering can
        clean up after the fact, since it's indistinguishable from a well-matched point until you
        triangulate it. Rather than render a cloud that looks broken for reasons that have nothing to do
        with the matching itself, this demo checks the median parallax angle up front and stops with an
        explanation if it's too low. Use the sample button below for a known-good pair.
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
            {sampleLoading ? "Loading…" : "Use sample image pair"}
          </button>

          <button
            onClick={() => handleRun()}
            disabled={leftFile.length === 0 || rightFile.length === 0 || loading}
            className="w-full rounded-md bg-neutral-900 px-4 py-2 text-sm font-medium text-white transition disabled:opacity-40 dark:bg-neutral-100 dark:text-neutral-900"
          >
            {loading ? "Reconstructing…" : "Run reconstruction"}
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
          {result && <Stepper steps={buildSteps(result)} />}
        </div>
      </div>
    </div>
  );
}
