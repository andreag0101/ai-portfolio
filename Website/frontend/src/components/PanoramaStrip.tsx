interface PanoramaStripProps {
  src: string;
  height?: number;
}

/** A wide panorama shown at a cropped height inside a horizontally scrollable
 * strip -- drag/swipe (or scroll) to pan across it. Native scroll works the
 * same on touch and desktop, so this needs no pointer-tracking JS. */
export default function PanoramaStrip({ src, height = 260 }: PanoramaStripProps) {
  return (
    <div
      className="group relative w-full cursor-grab overflow-x-auto overflow-y-hidden rounded-lg border border-neutral-200 bg-neutral-950 active:cursor-grabbing dark:border-neutral-800"
      style={{ height }}
    >
      <img src={src} alt="Stitched panorama" className="h-full max-w-none" style={{ height }} />
      <span className="pointer-events-none absolute right-2 top-2 rounded-md bg-black/60 px-2 py-0.5 text-xs font-medium text-white opacity-0 transition-opacity group-hover:opacity-100">
        Drag to pan →
      </span>
    </div>
  );
}
