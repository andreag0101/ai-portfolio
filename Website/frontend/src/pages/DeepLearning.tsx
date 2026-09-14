import { Link } from "react-router-dom";
import DLProjectCard from "../components/dl/DLProjectCard";
import { dlProjects } from "../lib/deepLearningProjects";

export default function DeepLearning() {
  return (
    <div className="mx-auto max-w-5xl px-6 py-10">
      <Link to="/" className="text-sm text-neutral-500 hover:text-neutral-900 dark:hover:text-neutral-100">
        &larr; Portfolio
      </Link>
      <h1 className="mt-2 text-2xl font-semibold tracking-tight text-neutral-900 dark:text-neutral-100">
        Deep Learning &mdash; ECE 60146
      </h1>
      <p className="mt-2 max-w-2xl text-sm text-neutral-600 dark:text-neutral-400">
        Coursework from Purdue's ECE 60146 (Deep Learning), compiled into eight projects &mdash; from
        hand-derived backpropagation to object detection, segmentation, GANs vs. diffusion, and
        Transformer translation. These are write-ups of results already trained and evaluated during
        the course, with the original figures, tables, and full PDF reports &mdash; not a live demo, so
        nothing here runs inference in your browser.
      </p>

      <div className="mt-8 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {dlProjects.map((p) => (
          <DLProjectCard key={p.slug} project={p} />
        ))}
      </div>
    </div>
  );
}
