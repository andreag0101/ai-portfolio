import { useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import { Link } from "react-router-dom";
import { rayHitsSegment, triangleEdges, type Vec2 } from "../lib/geometry";

// World coordinate bounds mapped onto the SVG viewport.
const WORLD = { xMin: -2, xMax: 10, yMin: -5, yMax: 7 };
const SIZE = 560;
const scale = SIZE / (WORLD.xMax - WORLD.xMin);

function worldToPixel([x, y]: Vec2): Vec2 {
  return [(x - WORLD.xMin) * scale, (WORLD.yMax - y) * scale];
}

function pixelToWorld(px: number, py: number): Vec2 {
  return [px / scale + WORLD.xMin, WORLD.yMax - py / scale];
}

type DragTarget = "p1" | "p2" | "p3" | "aim" | null;

export default function ProjectiveGeometry() {
  const [p1, setP1] = useState<Vec2>([3, 5]);
  const [p2, setP2] = useState<Vec2>([5, 3]);
  const [p3, setP3] = useState<Vec2>([7, 5]);
  const [aimHandle, setAimHandle] = useState<Vec2>([8, 4]);
  const dragging = useRef<DragTarget>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  const origin: Vec2 = [0, 0];
  const dir: Vec2 = normalize([aimHandle[0] - origin[0], aimHandle[1] - origin[1]]);
  const angleDeg = (Math.atan2(dir[1], dir[0]) * 180) / Math.PI;

  const edges = triangleEdges(p1, p2, p3);
  const hits = edges.map(([a, b]) => rayHitsSegment(origin, dir, a, b)).filter((h): h is NonNullable<typeof h> => h !== null);
  // A ray from outside the triangle that hits it crosses the boundary twice
  // (enter + exit); a ray starting inside the triangle crosses once (exit only).
  const sortedHits = [...hits].sort((a, b) => a.t - b.t);
  const hit = sortedHits.length >= 1;

  function setters(target: DragTarget) {
    if (target === "p1") return setP1;
    if (target === "p2") return setP2;
    if (target === "p3") return setP3;
    return setAimHandle;
  }

  function handlePointerDown(target: DragTarget) {
    return (e: ReactPointerEvent) => {
      dragging.current = target;
      (e.target as Element).setPointerCapture(e.pointerId);
    };
  }

  function handlePointerMove(e: ReactPointerEvent<SVGSVGElement>) {
    if (!dragging.current || !svgRef.current) return;
    const rect = svgRef.current.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * SIZE;
    const py = ((e.clientY - rect.top) / rect.height) * SIZE;
    const world = pixelToWorld(px, py);
    setters(dragging.current)(world);
  }

  function handlePointerUp() {
    dragging.current = null;
  }

  const rayFarPx = worldToPixel([origin[0] + dir[0] * 20, origin[1] + dir[1] * 20]);
  const originPx = worldToPixel(origin);

  return (
    <div className="mx-auto max-w-5xl px-6 py-10">
      <Link to="/" className="text-sm text-neutral-500 hover:text-neutral-900 dark:hover:text-neutral-100">
        &larr; All projects
      </Link>
      <h1 className="mt-2 text-2xl font-semibold tracking-tight">Projective Geometry Playground</h1>
      <p className="mt-2 max-w-2xl text-sm text-neutral-600 dark:text-neutral-400">
        Point-line duality: a line through two points, or the intersection of two lines, is just the
        cross product of their homogeneous coordinates. Drag the triangle's corners or the aim handle
        &mdash; everything recomputes live using that one operation.
      </p>

      <div className="mt-8 grid grid-cols-1 gap-8 lg:grid-cols-[1fr_260px]">
        <div className="rounded-lg border border-neutral-200 bg-white p-2 dark:border-neutral-800 dark:bg-neutral-900">
          <svg
            ref={svgRef}
            viewBox={`0 0 ${SIZE} ${SIZE}`}
            className="w-full touch-none select-none"
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
          >
            {/* grid */}
            {Array.from({ length: 13 }, (_, i) => WORLD.xMin + i).map((gx) => (
              <line
                key={`gx${gx}`}
                x1={worldToPixel([gx, WORLD.yMin])[0]}
                y1={0}
                x2={worldToPixel([gx, WORLD.yMin])[0]}
                y2={SIZE}
                stroke="currentColor"
                strokeOpacity={0.06}
                className="text-neutral-900 dark:text-neutral-100"
              />
            ))}
            {Array.from({ length: 13 }, (_, i) => WORLD.yMin + i).map((gy) => (
              <line
                key={`gy${gy}`}
                x1={0}
                y1={worldToPixel([WORLD.xMin, gy])[1]}
                x2={SIZE}
                y2={worldToPixel([WORLD.xMin, gy])[1]}
                stroke="currentColor"
                strokeOpacity={0.06}
                className="text-neutral-900 dark:text-neutral-100"
              />
            ))}

            {/* triangle */}
            <polygon
              points={[p1, p2, p3].map((p) => worldToPixel(p).join(",")).join(" ")}
              fill={hit ? "#10b98133" : "#3b82f633"}
              stroke={hit ? "#10b981" : "#3b82f6"}
              strokeWidth={2}
            />

            {/* ray */}
            <line
              x1={originPx[0]}
              y1={originPx[1]}
              x2={rayFarPx[0]}
              y2={rayFarPx[1]}
              stroke="#ef4444"
              strokeWidth={2}
              strokeDasharray="6 5"
            />
            <circle cx={originPx[0]} cy={originPx[1]} r={5} fill="#ef4444" />

            {/* intersection points */}
            {sortedHits.map((h, i) => {
              const px = worldToPixel(h.point);
              return <circle key={i} cx={px[0]} cy={px[1]} r={7} fill="none" stroke="#ef4444" strokeWidth={2.5} />;
            })}

            {/* draggable vertices */}
            {([
              ["p1", p1, "#3b82f6"],
              ["p2", p2, "#3b82f6"],
              ["p3", p3, "#3b82f6"],
              ["aim", aimHandle, "#ef4444"],
            ] as [DragTarget, Vec2, string][]).map(([target, p, color]) => {
              const px = worldToPixel(p);
              return (
                <circle
                  key={target}
                  cx={px[0]}
                  cy={px[1]}
                  r={9}
                  fill={color}
                  stroke="white"
                  strokeWidth={2}
                  className="cursor-grab active:cursor-grabbing"
                  onPointerDown={handlePointerDown(target)}
                />
              );
            })}
          </svg>
        </div>

        <div className="space-y-4">
          <div className={`rounded-lg border p-4 ${hit ? "border-gold-300 bg-gold-50 dark:border-gold-800 dark:bg-gold-950/30" : "border-neutral-200 bg-neutral-50 dark:border-neutral-800 dark:bg-neutral-900"}`}>
            <p className="text-xs font-medium uppercase tracking-wide text-neutral-500">Result</p>
            <p className={`mt-1 text-lg font-semibold ${hit ? "text-gold-700 dark:text-gold-400" : "text-neutral-700 dark:text-neutral-300"}`}>
              {hit ? "Aim is correct" : "Aim misses"}
            </p>
            <p className="mt-1 text-sm text-neutral-500">&alpha; = {angleDeg.toFixed(1)}&deg;</p>
          </div>

          <div className="space-y-1 text-sm text-neutral-600 dark:text-neutral-400">
            <p className="font-medium text-neutral-700 dark:text-neutral-300">How it works</p>
            <p>Each triangle edge and the aim ray are represented as homogeneous lines (cross product of their two endpoints).</p>
            <p>An intersection point is the cross product of two lines &mdash; the dual operation.</p>
            <p>The ray hits the triangle when it crosses exactly two of its three edges within their segment bounds.</p>
          </div>
        </div>
      </div>
    </div>
  );
}

function normalize([x, y]: Vec2): Vec2 {
  const len = Math.hypot(x, y) || 1;
  return [x / len, y / len];
}
