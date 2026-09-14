import { Link, Navigate, useParams } from "react-router-dom";
import ResultImage from "../components/ResultImage";
import CodeBlock from "../components/dl/CodeBlock";
import DataTable from "../components/dl/DataTable";
import { getDLProject } from "../lib/deepLearningProjects";

export default function DeepLearningProject() {
  const { slug } = useParams<{ slug: string }>();
  const project = slug ? getDLProject(slug) : undefined;

  if (!project) return <Navigate to="/deep-learning" replace />;

  return (
    <div className="mx-auto max-w-4xl px-6 py-10">
      <Link to="/deep-learning" className="text-sm text-neutral-500 hover:text-neutral-900 dark:hover:text-neutral-100">
        &larr; All Deep Learning projects
      </Link>

      <div className="mt-2 flex items-center gap-2">
        <span className="rounded-full bg-neutral-100 px-2 py-0.5 text-xs font-medium text-neutral-500 dark:bg-neutral-800 dark:text-neutral-400">
          ECE 60146 &middot; {project.hw}
        </span>
      </div>
      <h1 className="mt-2 text-2xl font-semibold tracking-tight text-neutral-900 dark:text-neutral-100">{project.title}</h1>

      {project.highlights.length > 0 && (
        <div className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-3">
          {project.highlights.map((h) => (
            <div key={h.label} className="rounded-lg border border-gold-300 bg-gold-50 px-3 py-2.5 dark:border-gold-800 dark:bg-gold-950/30">
              <p className="text-xs text-neutral-500 dark:text-neutral-400">{h.label}</p>
              <p className="mt-0.5 text-sm font-semibold text-gold-700 dark:text-gold-400">{h.value}</p>
            </div>
          ))}
        </div>
      )}

      <div className="mt-6 space-y-3 text-sm text-neutral-600 dark:text-neutral-400">
        {project.intro.map((p, i) => (
          <p key={i}>{p}</p>
        ))}
      </div>

      <div className="mt-10 space-y-12">
        {project.sections.map((section) => (
          <section key={section.heading}>
            <h2 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">{section.heading}</h2>

            {section.body && (
              <div className="mt-2 space-y-3 text-sm text-neutral-600 dark:text-neutral-400">
                {section.body.map((p, i) => (
                  <p key={i}>{p}</p>
                ))}
              </div>
            )}

            {section.images && (
              <div className={`mt-4 grid grid-cols-1 gap-4 ${section.images.length > 1 ? "sm:grid-cols-2" : ""}`}>
                {section.images.map((img) =>
                  img.narrow ? (
                    <div key={img.src} className="mx-auto w-full max-w-[280px]">
                      <ResultImage src={img.src} label={img.caption} />
                    </div>
                  ) : (
                    <ResultImage key={img.src} src={img.src} label={img.caption} />
                  ),
                )}
              </div>
            )}

            {section.table && (
              <div className="mt-4">
                <DataTable table={section.table} />
              </div>
            )}

            {section.code && (
              <div className="mt-4">
                <CodeBlock code={section.code} />
              </div>
            )}
          </section>
        ))}
      </div>

      <div className="mt-12 flex flex-wrap gap-3 border-t border-neutral-200 pt-6 dark:border-neutral-800">
        {project.reports.map((r) => (
          <a
            key={r.href}
            href={r.href}
            target="_blank"
            rel="noreferrer"
            className="rounded-md bg-neutral-900 px-4 py-2 text-sm font-medium text-white transition hover:bg-neutral-700 dark:bg-neutral-100 dark:text-neutral-900 dark:hover:bg-neutral-300"
          >
            {r.label} &rarr;
          </a>
        ))}
      </div>
    </div>
  );
}
