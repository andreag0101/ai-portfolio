import type { DLCode } from "../../lib/deepLearningProjects";

export default function CodeBlock({ code }: { code: DLCode }) {
  return (
    <div className="overflow-hidden rounded-lg border border-neutral-200 dark:border-neutral-800">
      <div className="border-b border-neutral-200 bg-neutral-50 px-3 py-1.5 text-xs font-medium text-neutral-600 dark:border-neutral-800 dark:bg-neutral-900 dark:text-neutral-400">
        {code.label}
      </div>
      <pre className="overflow-x-auto bg-neutral-900 p-4 text-xs leading-relaxed text-neutral-200">
        <code>{code.code}</code>
      </pre>
    </div>
  );
}
