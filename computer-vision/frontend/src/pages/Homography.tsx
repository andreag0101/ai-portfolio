import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import FilePicker from "../components/FilePicker";
import ResultImage from "../components/ResultImage";
import QuadPicker from "../components/QuadPicker";
import Stepper, { type Step } from "../components/Stepper";
import {
  runDewarp,
  runInsert,
  getRectifySample,
  getRectifyInsertSample,
  dataUrlToFile,
  type DewarpResult,
  type InsertResult,
  type Quad,
} from "../lib/api";

function dewarpSteps(result: DewarpResult): Step[] {
  return [
    {
      title: "Chosen region",
      description: "The quadrilateral picked above (automatically or by hand), outlined on the source photo.",
      content: <ResultImage label="Chosen region" src={result.detectedQuad} />,
    },
    {
      title: "Straightened",
      description: "A 4-point DLT homography maps the quad's corners to a fronto-parallel rectangle, then every source pixel is forward-warped into it.",
      content: <ResultImage label="Straightened" src={result.rectified} />,
    },
  ];
}

function insertSteps(result: InsertResult): Step[] {
  return [
    {
      title: "Chosen frame",
      description: "The quadrilateral picked above, outlined on the destination photo.",
      content: <ResultImage label="Chosen frame" src={result.detectedQuad} />,
    },
    {
      title: "Composited",
      description: "The uploaded image's four corners are mapped onto the quad's corners via the same DLT homography, then warped in.",
      content: <ResultImage label="Composited" src={result.composited} />,
    },
  ];
}

type Tab = "straighten" | "insert";

export default function Homography() {
  const [tab, setTab] = useState<Tab>("straighten");

  return (
    <div className="mx-auto max-w-5xl px-6 py-10">
      <Link to="/" className="text-sm text-neutral-500 hover:text-neutral-900 dark:hover:text-neutral-100">
        &larr; All projects
      </Link>
      <h1 className="mt-2 text-2xl font-semibold tracking-tight">Planar Rectification & Compositing</h1>
      <p className="mt-2 max-w-2xl text-sm text-neutral-600 dark:text-neutral-400">
        Both tools automatically find candidate quadrilaterals in a photo &mdash; Canny edges, contour
        detection, and polygon approximation, the same approach document scanners use &mdash; then solve
        a direct linear transform homography from the four corner correspondences to warp the image. No
        clicking corners by hand. Photos with nested rectangles (a frame's outer edge, its mat, and the
        picture inside it) or partial occlusion are genuinely ambiguous, so pick from the detected options
        if the first guess isn't the one you meant.
      </p>

      <div className="mt-6 flex gap-2">
        <button
          onClick={() => setTab("straighten")}
          className={`rounded-md px-3 py-1.5 text-sm ${
            tab === "straighten"
              ? "bg-neutral-900 text-white dark:bg-neutral-100 dark:text-neutral-900"
              : "bg-neutral-100 text-neutral-700 dark:bg-neutral-800 dark:text-neutral-300"
          }`}
        >
          Straighten a photo
        </button>
        <button
          onClick={() => setTab("insert")}
          className={`rounded-md px-3 py-1.5 text-sm ${
            tab === "insert"
              ? "bg-neutral-900 text-white dark:bg-neutral-100 dark:text-neutral-900"
              : "bg-neutral-100 text-neutral-700 dark:bg-neutral-800 dark:text-neutral-300"
          }`}
        >
          Insert into a frame
        </button>
      </div>

      {tab === "straighten" ? <Straighten /> : <Insert />}
    </div>
  );
}

function Straighten() {
  const [files, setFiles] = useState<File[]>([]);
  const [quad, setQuad] = useState<Quad | null>(null);
  const [loading, setLoading] = useState(false);
  const [sampleLoading, setSampleLoading] = useState(false);
  const [autoRun, setAutoRun] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<DewarpResult | null>(null);

  async function handleRun() {
    if (files.length === 0 || !quad) return;
    setLoading(true);
    setError(null);
    try {
      setResult(await runDewarp(files[0], quad));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (autoRun && quad && files.length > 0) {
      setAutoRun(false);
      handleRun();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoRun, quad, files]);

  async function handleUseSample() {
    setSampleLoading(true);
    setError(null);
    try {
      const { image } = await getRectifySample();
      const file = await dataUrlToFile(image, "laptop-screen.jpg");
      setFiles([file]);
      setResult(null);
      setAutoRun(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong");
    } finally {
      setSampleLoading(false);
    }
  }

  return (
    <div className="mt-6 grid grid-cols-1 gap-8 lg:grid-cols-[280px_1fr]">
      <div className="space-y-5">
        <p className="text-sm text-neutral-600 dark:text-neutral-400">
          Upload a photo of a document, sign, screen, or picture frame shot at an angle. The plane's
          edges should contrast clearly with the background.
        </p>
        <FilePicker
          label="Photo"
          files={files}
          onChange={(f) => {
            setFiles(f);
            setResult(null);
          }}
        />
        <button
          onClick={handleUseSample}
          disabled={sampleLoading || loading}
          className="w-full rounded-md bg-neutral-100 px-4 py-2 text-sm font-medium text-neutral-700 transition disabled:opacity-40 dark:bg-neutral-800 dark:text-neutral-300"
        >
          {sampleLoading ? "Loading…" : "Use sample photo"}
        </button>
        <QuadPicker file={files[0] ?? null} onQuadChange={setQuad} />
        <button
          onClick={handleRun}
          disabled={files.length === 0 || !quad || loading}
          className="w-full rounded-md bg-neutral-900 px-4 py-2 text-sm font-medium text-white transition disabled:opacity-40 dark:bg-neutral-100 dark:text-neutral-900"
        >
          {loading ? "Straightening…" : "Straighten"}
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
        {result && <Stepper steps={dewarpSteps(result)} />}
      </div>
    </div>
  );
}

function Insert() {
  const [destFiles, setDestFiles] = useState<File[]>([]);
  const [sourceFiles, setSourceFiles] = useState<File[]>([]);
  const [quad, setQuad] = useState<Quad | null>(null);
  const [loading, setLoading] = useState(false);
  const [sampleLoading, setSampleLoading] = useState(false);
  const [autoRun, setAutoRun] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<InsertResult | null>(null);

  async function handleRun() {
    if (destFiles.length === 0 || sourceFiles.length === 0 || !quad) return;
    setLoading(true);
    setError(null);
    try {
      setResult(await runInsert(destFiles[0], sourceFiles[0], quad));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (autoRun && quad && destFiles.length > 0 && sourceFiles.length > 0) {
      setAutoRun(false);
      handleRun();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoRun, quad, destFiles, sourceFiles]);

  async function handleUseSample() {
    setSampleLoading(true);
    setError(null);
    try {
      const { dest, source } = await getRectifyInsertSample();
      const [destFile, sourceFile] = await Promise.all([
        dataUrlToFile(dest, "picture-frame.jpeg"),
        dataUrlToFile(source, "alex-honnold.jpg"),
      ]);
      setDestFiles([destFile]);
      setSourceFiles([sourceFile]);
      setResult(null);
      setAutoRun(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong");
    } finally {
      setSampleLoading(false);
    }
  }

  return (
    <div className="mt-6 grid grid-cols-1 gap-8 lg:grid-cols-[280px_1fr]">
      <div className="space-y-5">
        <p className="text-sm text-neutral-600 dark:text-neutral-400">
          Upload a destination photo containing a blank rectangular region (a screen, picture frame,
          poster, wall panel), and a source image to warp into it.
        </p>
        <FilePicker
          label="Destination photo (with a frame)"
          files={destFiles}
          onChange={(f) => {
            setDestFiles(f);
            setResult(null);
          }}
        />
        <button
          onClick={handleUseSample}
          disabled={sampleLoading || loading}
          className="w-full rounded-md bg-neutral-100 px-4 py-2 text-sm font-medium text-neutral-700 transition disabled:opacity-40 dark:bg-neutral-800 dark:text-neutral-300"
        >
          {sampleLoading ? "Loading…" : "Use sample photos"}
        </button>
        <QuadPicker file={destFiles[0] ?? null} onQuadChange={setQuad} />
        <FilePicker label="Image to insert" files={sourceFiles} onChange={setSourceFiles} />
        <button
          onClick={handleRun}
          disabled={destFiles.length === 0 || sourceFiles.length === 0 || !quad || loading}
          className="w-full rounded-md bg-neutral-900 px-4 py-2 text-sm font-medium text-white transition disabled:opacity-40 dark:bg-neutral-100 dark:text-neutral-900"
        >
          {loading ? "Compositing…" : "Insert"}
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
        {result && <Stepper steps={insertSteps(result)} />}
      </div>
    </div>
  );
}
