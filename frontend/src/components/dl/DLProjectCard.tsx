import { Link } from "react-router-dom";
import type { DLProject } from "../../lib/deepLearningProjects";

export default function DLProjectCard({ project }: { project: DLProject }) {
  return (
    <Link to={`/deep-learning/${project.slug}`} className="block h-full">
      <div className="group h-full overflow-hidden rounded-xl border border-neutral-200 bg-white transition hover:border-neutral-300 hover:shadow-md dark:border-neutral-800 dark:bg-neutral-900 dark:hover:border-neutral-700">
        <img src={project.thumbnail} alt="" className="h-32 w-full border-b border-neutral-200 object-cover dark:border-neutral-800" />
        <div className="p-5">
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
      </div>
    </Link>
  );
}
