import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import FilePicker from "../components/FilePicker";
import ResultImage from "../components/ResultImage";
import Stepper, { type Step } from "../components/Stepper";
import { runPanorama, getPanoramaSample, getPrecomputed, dataUrlToFile, type PanoramaResult } from "../lib/api";

function buildSteps(result: PanoramaResult): Step[] {
  return [
    {
      title: "Interest points",
      description: "SIFT keypoints detected independently in each photo (circle size = scale, radial line = orientation).",
      content: (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          {result.keypointOverlays.map((src, i) => (
            <ResultImage key={i} label={`Image ${i + 1}`} src={src} />
          ))}
        </div>
      ),
    },
    {
      title: "Matching",
      description: "Each consecutive pair matched with a ratio test, then a from-scratch RANSAC separates inliers (green) from outliers (red) for the homography fit.",
      content: (
        <div className="space-y-3">
          {result.matchOverlays.map((src, i) => (
            <ResultImage key={i} label={`Image ${i + 1} ↔ Image ${i + 2}`} src={src} />
          ))}
        </div>
      ),
    },
    {
      title: "Stitched panorama",
      description: "Every image warped into the middle photo's frame via composed homographies, refined with Levenberg-Marquardt.",
      content: <ResultImage label="Stitched panorama" src={result.panorama} />,
    },
  ];
}

export default function Panorama() {
  const [files, setFiles] = useState<File[]>([]);
  const [ransacConst, setRansacConst] = useState(10);
  const [loading, setLoading] = useState(false);
  const [sampleLoading, setSampleLoading] = useState(false);
  const [statusMsg, setStatusMsg] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PanoramaResult | null>(null);
  const [isDefault, setIsDefault] = useState(false);

  useEffect(() => {
    getPrecomputed<PanoramaResult>("panorama")
      .then((r) => {
        setResult(r);
        setIsDefault(true);
      })
      .catch(() => {});
  }, []);

  async function handleRun(useFiles?: File[], useConst?: number) {
    const fs = useFiles ?? files;
    if (fs.length < 2) return;
    setLoading(true);
    setError(null);
    setStatusMsg("Matching features and stitching — this can take a few seconds…");
    try {
      setResult(await runPanorama(fs, useConst ?? ransacConst));
      setIsDefault(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong");
    } finally {
      setLoading(false);
      setStatusMsg(null);
    }
  }

  async function handleUseSample() {
    setSampleLoading(true);
    setError(null);
    try {
      const { images } = await getPanoramaSample();
      const sampleFiles = await Promise.all(images.map((dataUrl, i) => dataUrlToFile(dataUrl, `fountain-${i + 1}.jpg`)));
      setFiles(sampleFiles);
      // A tighter tolerance than the default gives a visibly cleaner stitch for this
      // sequence specifically -- the fountain is a close foreground object, so wider
      // tolerances let genuinely inconsistent (parallax-affected) matches into the
      // homography fit.
      setRansacConst(5);
      await handleRun(sampleFiles, 5);
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
      <h1 className="mt-2 text-2xl font-semibold tracking-tight">Panorama Stitching</h1>
      <p className="mt-2 max-w-2xl text-sm text-neutral-600 dark:text-neutral-400">
        Upload 2&ndash;6 overlapping photos, left to right. Each pair is matched with SIFT, a
        homography is estimated with a from-scratch RANSAC, refined with Levenberg-Marquardt, and
        every image is warped into the frame of the middle photo. Straight-line perspective warping
        (no cylindrical projection) means wide sequences can bow outward at the edges &mdash; that's
        expected, not a bug.
      </p>

      <div className="mt-8 grid grid-cols-1 gap-8 lg:grid-cols-[280px_1fr]">
        <div className="space-y-5">
          <FilePicker label="Images (in order, left to right)" files={files} onChange={setFiles} multiple />

          <button
            onClick={handleUseSample}
            disabled={sampleLoading || loading}
            className="w-full rounded-md bg-neutral-100 px-4 py-2 text-sm font-medium text-neutral-700 transition disabled:opacity-40 dark:bg-neutral-800 dark:text-neutral-300"
          >
            {sampleLoading ? "Loading…" : "Use sample photo sequence"}
          </button>

          <div>
            <label className="mb-1.5 flex justify-between text-sm font-medium text-neutral-700 dark:text-neutral-300">
              <span>RANSAC inlier tolerance</span>
              <span className="text-neutral-400">{ransacConst}</span>
            </label>
            <input
              type="range"
              min={2}
              max={30}
              value={ransacConst}
              onChange={(e) => setRansacConst(Number(e.target.value))}
              className="w-full"
            />
          </div>

          <button
            onClick={() => handleRun()}
            disabled={files.length < 2 || loading}
            className="w-full rounded-md bg-neutral-900 px-4 py-2 text-sm font-medium text-white transition disabled:opacity-40 dark:bg-neutral-100 dark:text-neutral-900"
          >
            {loading ? "Stitching…" : "Stitch panorama"}
          </button>

          {statusMsg && <p className="text-sm text-neutral-500">{statusMsg}</p>}
          {error && <p className="text-sm text-red-600 dark:text-red-400">{error}</p>}
        </div>

        <div>
          {!result && !loading && (
            <div className="flex h-64 items-center justify-center rounded-lg border border-dashed border-neutral-300 text-sm text-neutral-400 dark:border-neutral-700">
              Result will appear here
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
                  Showing the bundled sample result. Upload your own photo sequence to run it live.
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
