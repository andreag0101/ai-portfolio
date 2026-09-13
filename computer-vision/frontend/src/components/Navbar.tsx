import { Link } from "react-router-dom";

export default function Navbar() {
  return (
    <header className="border-b border-neutral-200 dark:border-neutral-800">
      <div className="mx-auto flex max-w-5xl items-center justify-between px-6 py-4">
        <Link to="/" className="text-lg font-semibold tracking-tight">
          Andrea Goh <span className="text-neutral-400 dark:text-neutral-600">/</span> Computer Vision
        </Link>
        <nav className="flex gap-5 text-sm text-neutral-600 dark:text-neutral-400">
          <Link to="/" className="hover:text-neutral-900 dark:hover:text-neutral-100">
            All demos
          </Link>
        </nav>
      </div>
    </header>
  );
}
