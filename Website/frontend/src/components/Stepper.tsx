import { useState, type ReactNode } from "react";

export interface Step {
  title: string;
  description?: string;
  content: ReactNode;
}

/**
 * Presents a pipeline's stages either one at a time (with Previous/Next and
 * clickable step pills) or all at once, stacked. Every demo's algorithm is a
 * sequence of stages under the hood; this is the one place that turns any of
 * them into a click-through walkthrough instead of a single final image.
 */
export default function Stepper({ steps }: { steps: Step[] }) {
  const [mode, setMode] = useState<"step" | "all">("step");
  const [idx, setIdx] = useState(0);

  if (steps.length === 0) return null;
  const step = steps[Math.min(idx, steps.length - 1)];

  return (
    <div>
      <div className="mb-3 flex items-center justify-between">
        <div className="flex flex-wrap gap-1.5">
          {mode === "step" &&
            steps.map((s, i) => (
              <button
                key={i}
                onClick={() => setIdx(i)}
                className={`flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs transition ${
                  i === idx
                    ? "bg-neutral-900 text-white dark:bg-neutral-100 dark:text-neutral-900"
                    : "bg-neutral-100 text-neutral-600 hover:bg-neutral-200 dark:bg-neutral-800 dark:text-neutral-400 dark:hover:bg-neutral-700"
                }`}
              >
                <span
                  className={`flex h-4 w-4 items-center justify-center rounded-full text-[10px] ${
                    i === idx ? "bg-white/20" : "bg-neutral-300 dark:bg-neutral-700"
                  }`}
                >
                  {i + 1}
                </span>
                {s.title}
              </button>
            ))}
        </div>
        <div className="flex shrink-0 gap-1 text-xs">
          <button
            onClick={() => setMode("step")}
            className={`rounded-md px-2 py-1 ${mode === "step" ? "bg-neutral-200 dark:bg-neutral-700" : "text-neutral-400"}`}
          >
            Step-by-step
          </button>
          <button
            onClick={() => setMode("all")}
            className={`rounded-md px-2 py-1 ${mode === "all" ? "bg-neutral-200 dark:bg-neutral-700" : "text-neutral-400"}`}
          >
            All at once
          </button>
        </div>
      </div>

      {mode === "step" ? (
        <div>
          <h3 className="text-sm font-semibold text-neutral-900 dark:text-neutral-100">{step.title}</h3>
          {step.description && <p className="mt-1 text-sm text-neutral-500">{step.description}</p>}
          <div className="mt-3">{step.content}</div>
          <div className="mt-4 flex items-center justify-between border-t border-neutral-200 pt-3 dark:border-neutral-800">
            <button
              onClick={() => setIdx((i) => Math.max(0, i - 1))}
              disabled={idx === 0}
              className="text-sm text-neutral-600 disabled:opacity-30 dark:text-neutral-400"
            >
              &larr; Previous
            </button>
            <span className="text-xs text-neutral-400">
              Step {idx + 1} of {steps.length}
            </span>
            <button
              onClick={() => setIdx((i) => Math.min(steps.length - 1, i + 1))}
              disabled={idx === steps.length - 1}
              className="text-sm text-neutral-600 disabled:opacity-30 dark:text-neutral-400"
            >
              Next &rarr;
            </button>
          </div>
        </div>
      ) : (
        <div className="space-y-8">
          {steps.map((s, i) => (
            <div key={i}>
              <h3 className="text-sm font-semibold text-neutral-900 dark:text-neutral-100">
                {i + 1}. {s.title}
              </h3>
              {s.description && <p className="mt-1 text-sm text-neutral-500">{s.description}</p>}
              <div className="mt-3">{s.content}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
