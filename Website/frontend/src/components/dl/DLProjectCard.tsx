import { Link } from "react-router-dom";
import type { DLProject } from "../../lib/deepLearningProjects";

export default function DLProjectCard({ project }: { project: DLProject }) {
  return (
    <Link to={`/deep-learning/${project.slug}`} className="block h-full">
      <div className="group h-full rounded-xl border border-neutral-200 bg-white p-5 transition hover:border-neutral-300 hover:shadow-md dark:border-neutral-800 dark:bg-neutral-900 dark:hover:border-neutral-700">
        <div className="mb-2 flex items-center justify-between gap-2">
          <span className="rounded-full bg-neutral-100 px-2 py-0.5 text-xs font-medium text-neutral-500 dark:bg-neutral-800 dark:text-neutral-400">
            {project.hw}
          </span>
          <span className="rounded-full bg-gold-100 px-2 py-0.5 text-xs font-medium text-gold-700 dark:bg-gold-900/40 dark:text-gold-400">
            Report
          </span>
        </div>
        <h3 className="text-base font-semibold text-neutral-900 dark:text-neutral-100">{project.title}</h3>
        <p className="mt-1.5 text-sm text-neutral-600 dark:text-neutral-400">{project.blurb}</p>
        <div className="mt-3 flex flex-wrap gap-1.5">
          {project.topics.map((t) => (
            <span key={t} className="rounded-md bg-neutral-100 px-2 py-0.5 text-xs text-neutral-600 dark:bg-neutral-800 dark:text-neutral-400">
              {t}
            </span>
          ))}
        </div>
      </div>
    </Link>
  );
}
