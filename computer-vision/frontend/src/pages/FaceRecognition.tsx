import { useEffect, useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import FilePicker from "../components/FilePicker";
import ResultImage from "../components/ResultImage";
import AccuracyChart from "../components/AccuracyChart";
import Stepper, { type Step } from "../components/Stepper";
import {
  runFaceClassify,
  getFaceAccuracyCurve,
  getFaceSamples,
  getPrecomputed,
  dataUrlToFile,
  type FaceClassifyResult,
  type FaceAccuracyCurve,
} from "../lib/api";

function BasisRow({ images }: { images: string[] }) {
  return (
    <div className="flex flex-wrap gap-2">
      {images.map((src, i) => (
        <img key={i} src={src} alt={`Basis ${i + 1}`} className="h-20 w-20 rounded-md border border-neutral-200 object-cover dark:border-neutral-800" />
      ))}
    </div>
  );
}

function buildSteps(result: FaceClassifyResult, curve: ReactNode): Step[] {
  return [
    {
      title: "Input",
      description: "Resized to 128×128 grayscale, no face detection or alignment.",
      content: <ResultImage label="Input" src={result.resizedInput} />,
    },
    {
      title: "Eigenfaces & fisherfaces",
      description: "The first 6 basis directions of each subspace, reshaped back into images. A face is described by how much of each of these it contains, not by its raw pixels.",
      content: (
        <div className="space-y-4">
          <div>
            <p className="mb-1.5 text-xs font-medium text-neutral-500">PCA (eigenfaces)</p>
            <BasisRow images={result.eigenfaces} />
          </div>
          <div>
            <p className="mb-1.5 text-xs font-medium text-neutral-500">Fisher-LDA (fisherfaces)</p>
            <BasisRow images={result.fisherfaces} />
          </div>
        </div>
      ),
    },
    {
      title: "Nearest match",
      description: "1-nearest-neighbor in each subspace against the 630 training photos.",
      content: (
        <div className="space-y-6">
          <div className="grid grid-cols-2 gap-4">
            <MatchCard title="PCA (eigenfaces) match" match={result.pca} />
            <MatchCard title="LDA (fisherfaces) match" match={result.lda} />
          </div>
          {curve}
        </div>
      ),
    },
  ];
}

export default function FaceRecognition() {
  const [files, setFiles] = useState<File[]>([]);
  const [k, setK] = useState(10);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<FaceClassifyResult | null>(null);
  const [isDefault, setIsDefault] = useState(false);
  const [samples, setSamples] = useState<string[]>([]);
  const [curve, setCurve] = useState<FaceAccuracyCurve | null>(null);

  useEffect(() => {
    getFaceSamples().then((r) => setSamples(r.samples)).catch(() => {});
    getFaceAccuracyCurve().then(setCurve).catch(() => {});
    getPrecomputed<{ result: FaceClassifyResult; accuracyCurve: FaceAccuracyCurve }>("face")
      .then((r) => {
        setResult(r.result);
        setCurve((c) => c ?? r.accuracyCurve);
        setIsDefault(true);
      })
      .catch(() => {});
  }, []);

  async function handleRun(file: File) {
    setLoading(true);
    setError(null);
    try {
      setResult(await runFaceClassify(file, k));
      setIsDefault(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  async function handleSampleClick(dataUrl: string, i: number) {
    const file = await dataUrlToFile(dataUrl, `sample-${i}.png`);
    setFiles([file]);
    handleRun(file);
  }

  return (
    <div className="mx-auto max-w-5xl px-6 py-10">
      <Link to="/" className="text-sm text-neutral-500 hover:text-neutral-900 dark:hover:text-neutral-100">
        &larr; All projects
      </Link>
      <h1 className="mt-2 text-2xl font-semibold tracking-tight">Face Recognition: Eigenfaces vs. Fisherfaces</h1>
      <p className="mt-2 max-w-2xl text-sm text-neutral-600 dark:text-neutral-400">
        PCA (eigenfaces) and Fisher-LDA (fisherfaces) each project a face onto a low-dimensional subspace
        learned from 630 photos of 30 people, then find the nearest training photo in that subspace. This
        is <strong>closed-set recognition against exactly those 30 people</strong>, with no face detection
        or alignment step &mdash; your photo just gets resized to 128&times;128 and compared directly, so
        it will always return whichever of the 30 is closest, however good or bad that match really is. It
        works best with a similarly tightly-cropped, front-facing photo &mdash; or try one of the samples.
      </p>

      <div className="mt-8 grid grid-cols-1 gap-8 lg:grid-cols-[280px_1fr]">
        <div className="space-y-5">
          <FilePicker
            label="Your photo"
            files={files}
            onChange={(f) => {
              setFiles(f);
              if (f[0]) handleRun(f[0]);
            }}
          />

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
            <label className="mb-1.5 flex justify-between text-sm font-medium text-neutral-700 dark:text-neutral-300">
              <span>Subspace dimensions (K)</span>
              <span className="text-neutral-400">{k}</span>
            </label>
            <input
              type="range"
              min={1}
              max={29}
              value={k}
              onChange={(e) => {
                const val = Number(e.target.value);
                setK(val);
                if (files[0]) handleRun(files[0]);
              }}
              className="w-full"
            />
          </div>

          {error && <p className="text-sm text-red-600 dark:text-red-400">{error}</p>}
        </div>

        <div className="space-y-6">
          {!result && !loading && (
            <div className="flex h-48 items-center justify-center rounded-lg border border-dashed border-neutral-300 text-sm text-neutral-400 dark:border-neutral-700">
              Upload a photo or pick a sample
            </div>
          )}
          {loading && (
            <div className="flex h-48 items-center justify-center rounded-lg border border-dashed border-neutral-300 text-sm text-neutral-400 dark:border-neutral-700">
              Projecting…
            </div>
          )}
          {result && (
            <>
              {isDefault && (
                <p className="mb-3 text-sm text-neutral-500">
                  Showing the bundled sample result. Upload your own photo to run it live.
                </p>
              )}
              <Stepper
                steps={buildSteps(
                  result,
                  curve ? (
                    <div>
                      <p className="mb-2 text-sm font-medium text-neutral-700 dark:text-neutral-300">
                        Held-out test accuracy vs. K
                      </p>
                      <AccuracyChart k={curve.k} pca={curve.pca} lda={curve.lda} />
                    </div>
                  ) : null,
                )}
              />
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function MatchCard({ title, match }: { title: string; match: { label: number; distance: number; thumbnail: string } }) {
  return (
    <div className="rounded-lg border border-neutral-200 p-3 dark:border-neutral-800">
      <p className="text-xs font-medium text-neutral-500">{title}</p>
      <img src={match.thumbnail} alt={`Subject ${match.label}`} className="mt-2 w-full rounded-md" />
      <p className="mt-2 text-sm font-semibold text-neutral-900 dark:text-neutral-100">Subject {match.label}</p>
    </div>
  );
}
