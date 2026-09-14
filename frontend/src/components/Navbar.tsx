import { Link, useLocation } from "react-router-dom";

const SECTIONS = [
  { id: "about", label: "About" },
  { id: "projects", label: "Projects" },
  { id: "education", label: "Education" },
  { id: "career", label: "Career" },
];

export default function Navbar() {
  const { pathname } = useLocation();
  const onHome = pathname === "/";

  return (
    <header className="sticky top-0 z-10 border-b border-neutral-200 bg-white/80 backdrop-blur dark:border-neutral-800 dark:bg-neutral-950/80">
      <div className="mx-auto flex max-w-5xl items-center justify-between gap-3 px-6 py-4">
        <Link to="/" className="shrink-0 text-base font-semibold tracking-tight whitespace-nowrap text-neutral-900 dark:text-neutral-100 sm:text-lg">
          Andrea Goh
        </Link>
        <nav className="flex gap-3 text-sm text-neutral-600 dark:text-neutral-400 sm:gap-5">
          {onHome ? (
            SECTIONS.map((s) => (
              <a key={s.id} href={`#${s.id}`} className="hover:text-neutral-900 dark:hover:text-neutral-100">
                {s.label}
              </a>
            ))
          ) : (
            <Link to="/#projects" className="hover:text-neutral-900 dark:hover:text-neutral-100">
              &larr; Portfolio
            </Link>
          )}
        </nav>
      </div>
    </header>
  );
}
