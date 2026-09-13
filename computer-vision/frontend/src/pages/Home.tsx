import ProjectCard from "../components/ProjectCard";
import { categories } from "../lib/projects";

export default function Home() {
  return (
    <div className="mx-auto max-w-5xl px-6 py-12">
      <h1 className="text-3xl font-semibold tracking-tight text-neutral-900 dark:text-neutral-100">
        Computer Vision, from scratch
      </h1>
      <p className="mt-3 max-w-2xl text-neutral-600 dark:text-neutral-400">
        A collection of classical computer vision techniques, implemented from first principles &mdash;
        no relying on OpenCV's built-in solvers for the core algorithms. Pick a card marked{" "}
        <span className="font-medium text-emerald-700 dark:text-emerald-400">Try it</span> to run it on
        your own images.
      </p>

      <div className="mt-12 space-y-12">
        {categories.map((cat) => (
          <section key={cat.id}>
            <h2 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">{cat.title}</h2>
            <p className="mt-1 max-w-2xl text-sm text-neutral-500 dark:text-neutral-500">{cat.description}</p>
            <div className="mt-5 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {cat.projects.map((project) => (
                <ProjectCard key={project.id} project={project} />
              ))}
            </div>
          </section>
        ))}
      </div>
    </div>
  );
}
