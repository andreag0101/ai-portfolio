import { Link } from "react-router-dom";
import DLProjectCard from "../components/dl/DLProjectCard";
import { dlCategories, dlProjects } from "../lib/deepLearningProjects";

export default function DeepLearning() {
  return (
    <div className="mx-auto max-w-5xl px-6 py-10">
      <Link to="/" className="text-sm text-neutral-500 hover:text-neutral-900 dark:hover:text-neutral-100">
        &larr; Portfolio
      </Link>
      <h1 className="mt-2 text-2xl font-semibold tracking-tight text-neutral-900 dark:text-neutral-100">Deep Learning</h1>
      <p className="mt-2 max-w-2xl text-sm text-neutral-600 dark:text-neutral-400">
        A collection of deep learning projects, from hand-derived backpropagation to generative
        models, implemented and trained in PyTorch: from scratch where the point is understanding
        the mechanics, and on top of established architectures where the point is the result.
      </p>
      <p className="mt-3 max-w-2xl text-sm text-neutral-500 dark:text-neutral-500">
        Each project below shows the trained results: figures, tables, and a full technical
        write-up. Nothing here runs live inference in your browser.
      </p>

      <div className="mt-10 space-y-12">
        {dlCategories.map((cat) => {
          const projects = dlProjects.filter((p) => p.category === cat.id);
          if (projects.length === 0) return null;
          return (
            <div key={cat.id}>
              <h2 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">{cat.title}</h2>
              <p className="mt-1 max-w-2xl text-sm text-neutral-500 dark:text-neutral-500">{cat.description}</p>
              <div className="mt-5 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
                {projects.map((p) => (
                  <DLProjectCard key={p.slug} project={p} />
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
