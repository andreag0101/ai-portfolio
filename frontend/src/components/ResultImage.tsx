export default function ResultImage({ label, src }: { label: string; src: string }) {
  return (
    <figure className="overflow-hidden rounded-lg border border-neutral-200 dark:border-neutral-800">
      <img src={src} alt={label} className="w-full bg-neutral-900" />
      <figcaption className="border-t border-neutral-200 bg-neutral-50 px-3 py-1.5 text-xs text-neutral-600 dark:border-neutral-800 dark:bg-neutral-900 dark:text-neutral-400">
        {label}
      </figcaption>
    </figure>
  );
}
