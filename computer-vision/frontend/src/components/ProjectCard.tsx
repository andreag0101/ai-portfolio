import { Link } from "react-router-dom";
import type { ProjectMeta } from "../lib/projects";

export default function ProjectCard({ project }: { project: ProjectMeta }) {
  const live = Boolean(project.path);

  const content = (
    <div
      className={`group h-full rounded-xl border p-5 transition ${
        live
          ? "border-neutral-200 bg-white hover:border-neutral-300 hover:shadow-md dark:border-neutral-800 dark:bg-neutral-900 dark:hover:border-neutral-700"
          : "border-dashed border-neutral-200 bg-neutral-50/50 dark:border-neutral-800 dark:bg-neutral-900/30"
      }`}
    >
      <div className="mb-2 flex items-center justify-end">
        {live ? (
          <span className="rounded-full bg-emerald-100 px-2 py-0.5 text-xs font-medium text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-400">
            Try it
          </span>
        ) : (
          <span className="rounded-full bg-neutral-100 px-2 py-0.5 text-xs font-medium text-neutral-500 dark:bg-neutral-800 dark:text-neutral-400">
            Coming soon
          </span>
        )}
      </div>
      <h3 className="text-base font-semibold text-neutral-900 dark:text-neutral-100">{project.title}</h3>
      <p className="mt-1.5 text-sm text-neutral-600 dark:text-neutral-400">{project.blurb}</p>
      <div className="mt-3 flex flex-wrap gap-1.5">
        {project.topics.map((t) => (
          <span
            key={t}
            className="rounded-md bg-neutral-100 px-2 py-0.5 text-xs text-neutral-600 dark:bg-neutral-800 dark:text-neutral-400"
          >
            {t}
          </span>
        ))}
      </div>
    </div>
  );

  return live ? (
    <Link to={project.path!} className="block h-full">
      {content}
    </Link>
  ) : (
    content
  );
}
