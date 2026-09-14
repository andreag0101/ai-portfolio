import { useState } from "react";

interface FlipCardProps {
  front: string;
  back: string;
  frontLabel: string;
  backLabel: string;
  height?: number;
}

/** Click to flip between a "before" and "after" image. Each face keeps its own
 * natural aspect ratio (object-contain), so this works even when the two
 * images are differently shaped/cropped -- unlike an overlay slider, which
 * needs both images pixel-aligned to the same frame. */
export default function FlipCard({ front, back, frontLabel, backLabel, height = 260 }: FlipCardProps) {
  const [flipped, setFlipped] = useState(false);

  return (
    <button
      type="button"
      onClick={() => setFlipped((f) => !f)}
      className="group relative block w-full cursor-pointer overflow-hidden rounded-lg border border-neutral-200 bg-neutral-950 text-left dark:border-neutral-800"
      style={{ height, perspective: "1200px" }}
      aria-label="Click to flip between before and after"
    >
      <div
        className="relative h-full w-full transition-transform duration-700"
        style={{
          transformStyle: "preserve-3d",
          transform: flipped ? "rotateY(180deg)" : "rotateY(0deg)",
        }}
      >
        <div className="absolute inset-0 flex items-center justify-center" style={{ backfaceVisibility: "hidden" }}>
          <img src={front} alt={frontLabel} className="h-full w-full object-contain" />
          <span className="absolute bottom-2 left-2 rounded-md bg-black/60 px-2 py-0.5 text-xs font-medium text-white">
            {frontLabel}
          </span>
        </div>
        <div
          className="absolute inset-0 flex items-center justify-center"
          style={{ backfaceVisibility: "hidden", transform: "rotateY(180deg)" }}
        >
          <img src={back} alt={backLabel} className="h-full w-full object-contain" />
          <span className="absolute bottom-2 left-2 rounded-md bg-black/60 px-2 py-0.5 text-xs font-medium text-white">
            {backLabel}
          </span>
        </div>
      </div>
      <span className="pointer-events-none absolute right-2 top-2 rounded-md bg-black/60 px-2 py-0.5 text-xs font-medium text-white opacity-0 transition-opacity group-hover:opacity-100">
        Click to flip
      </span>
    </button>
  );
}
