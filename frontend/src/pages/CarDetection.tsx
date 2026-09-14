import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import FilePicker from "../components/FilePicker";
import ResultImage from "../components/ResultImage";
import Stepper, { type Step } from "../components/Stepper";
import { runCarDetection, getCarDetectionSamples, getPrecomputed, dataUrlToFile, type CarDetectionResult } from "../lib/api";

function buildSteps(result: CarDetectionResult): Step[] {
  const margin = result.score - result.threshold;
  return [
    {
      title: "Input patch",
      description: "Resized to 40×20. This classifies one already-cropped patch; it doesn't search a full photo for cars.",
      content: <ResultImage label="Input" src={result.resizedPatch} />,
    },
    {
      title: "Strongest features",
      description: "The 3 highest-weight Haar-like rectangles among the 20 boosted this round chose: each one sums pixels in the dark box minus the light box.",
      content: <ResultImage label="Top features" src={result.featureOverlay} />,
    },
    {
      title: "Classification",
      description: `A weighted vote across ${result.numRounds} boosted weak classifiers, compared to a threshold. On a held-out set of 618 test patches, this reaches ${(result.testAccuracy * 100).toFixed(1)}% accuracy.`,
      content: (
        <div className="space-y-4">
          <div className={`rounded-lg border p-4 ${result.isCar ? "border-gold-300 bg-gold-50 dark:border-gold-800 dark:bg-gold-950/30" : "border-neutral-200 bg-neutral-50 dark:border-neutral-800 dark:bg-neutral-900"}`}>
            <p className="text-xs font-medium uppercase tracking-wide text-neutral-500">Prediction</p>
            <p className={`mt-1 text-lg font-semibold ${result.isCar ? "text-gold-700 dark:text-gold-400" : "text-neutral-700 dark:text-neutral-300"}`}>
              {result.isCar ? "Car" : "Not a car"}
            </p>
            <p className="mt-1 text-xs text-neutral-500">
              score {result.score.toFixed(1)} vs. threshold {result.threshold.toFixed(1)} ({margin >= 0 ? "+" : ""}{margin.toFixed(1)})
            </p>
          </div>
          <div className="grid max-w-xs grid-cols-2 gap-x-6 gap-y-1 text-sm">
            <span className="text-neutral-500">True positive</span>
            <span className="font-mono">{result.confusion.tp}</span>
            <span className="text-neutral-500">False positive</span>
            <span className="font-mono">{result.confusion.fp}</span>
            <span className="text-neutral-500">False negative</span>
            <span className="font-mono">{result.confusion.fn}</span>
            <span className="text-neutral-500">True negative</span>
            <span className="font-mono">{result.confusion.tn}</span>
          </div>
        </div>
      ),
    },
  ];
}

export default function CarDetection() {
  const [files, setFiles] = useState<File[]>([]);
  const [loading, setLoading] = useState(false);
  const [statusMsg, setStatusMsg] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<CarDetectionResult | null>(null);
  const [isDefault, setIsDefault] = useState(false);
  const [samples, setSamples] = useState<{ positive: string[]; negative: string[] }>({ positive: [], negative: [] });

  useEffect(() => {
    getCarDetectionSamples().then(setSamples).catch(() => {});
    getPrecomputed<CarDetectionResult>("car-detection")
      .then((r) => {
        setResult(r);
        setIsDefault(true);
      })
      .catch(() => {});
  }, []);

  async function handleRun(file: File) {
    setLoading(true);
    setError(null);
    setStatusMsg("Classifying: the first request also trains the classifier, which can take about a minute…");
    try {
      setResult(await runCarDetection(file));
      setIsDefault(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong");
    } finally {
      setLoading(false);
      setStatusMsg(null);
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
      <h1 className="mt-2 text-2xl font-semibold tracking-tight">Car Detection: Haar Features + AdaBoost</h1>
      <p className="mt-2 max-w-2xl text-sm text-neutral-600 dark:text-neutral-400">
        Integral-image Haar-like features (11,200 of them, for a 40&times;20 patch) boosted with AdaBoost
        into a single strong classifier (the same feature family Viola-Jones face detection uses).
        This classifies one pre-cropped patch as car / not-car; it doesn't scan a full photo for cars
        (the original's multi-stage cascade exists for that kind of sliding-window search, which isn't
        what this demo does). Works best on a patch already cropped tightly around a vehicle or not, at
        roughly a 2:1 width:height ratio. Try a sample below for a guaranteed reasonable input.
      </p>

      <div className="mt-8 grid grid-cols-1 gap-8 lg:grid-cols-[280px_1fr]">
        <div className="space-y-5">
          <FilePicker
            label="Patch"
            files={files}
            onChange={(f) => {
              setFiles(f);
              if (f[0]) handleRun(f[0]);
            }}
          />

          {(samples.positive.length > 0 || samples.negative.length > 0) && (
            <div className="space-y-2">
              <p className="text-sm font-medium text-neutral-700 dark:text-neutral-300">Or try a sample</p>
              <div>
                <p className="mb-1 text-xs text-neutral-500">Cars</p>
                <div className="flex flex-wrap gap-1.5">
                  {samples.positive.map((s, i) => (
                    <button key={i} onClick={() => handleSampleClick(s, i)} className="overflow-hidden rounded-md border border-neutral-200 dark:border-neutral-800">
                      <img src={s} alt={`Positive sample ${i + 1}`} className="h-10 w-16 object-cover" />
                    </button>
                  ))}
                </div>
              </div>
              <div>
                <p className="mb-1 text-xs text-neutral-500">Not cars</p>
                <div className="flex flex-wrap gap-1.5">
                  {samples.negative.map((s, i) => (
                    <button key={i} onClick={() => handleSampleClick(s, i)} className="overflow-hidden rounded-md border border-neutral-200 dark:border-neutral-800">
                      <img src={s} alt={`Negative sample ${i + 1}`} className="h-10 w-16 object-cover" />
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}

          {statusMsg && <p className="text-sm text-neutral-500">{statusMsg}</p>}
          {error && <p className="text-sm text-red-600 dark:text-red-400">{error}</p>}
        </div>

        <div>
          {!result && !loading && (
            <div className="flex h-64 items-center justify-center rounded-lg border border-dashed border-neutral-300 text-sm text-neutral-400 dark:border-neutral-700">
              Upload a patch or pick a sample
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
                  Showing the bundled sample result. Upload your own patch to run it live.
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
