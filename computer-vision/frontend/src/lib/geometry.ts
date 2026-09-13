// Homogeneous point/line duality: a line through two points, or the
// intersection of two lines, is the cross product of their homogeneous
// coordinates. Ported from Homework 1's Question_7.py.

export type Vec2 = [number, number];
export type Vec3 = [number, number, number];

export function cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

export function toHomogeneous([x, y]: Vec2): Vec3 {
  return [x, y, 1];
}

export function lineThroughPoints(p: Vec2, q: Vec2): Vec3 {
  return cross(toHomogeneous(p), toHomogeneous(q));
}

/** Intersection of two homogeneous lines; null if parallel (point at infinity). */
export function intersectLines(l1: Vec3, l2: Vec3): Vec2 | null {
  const [x, y, w] = cross(l1, l2);
  if (Math.abs(w) < 1e-9) return null;
  return [x / w, y / w];
}

export interface RaySegmentHit {
  point: Vec2;
  t: number; // distance along the ray, in direction units
}

/**
 * Where a ray (from `origin`, unit direction `dir`) crosses the segment [a, b],
 * computed via the line duality above (edge line x ray line), then checked
 * against the segment's and ray's valid parameter ranges.
 */
export function rayHitsSegment(origin: Vec2, dir: Vec2, a: Vec2, b: Vec2): RaySegmentHit | null {
  const rayFar: Vec2 = [origin[0] + dir[0], origin[1] + dir[1]];
  const rayLine = lineThroughPoints(origin, rayFar);
  const edgeLine = lineThroughPoints(a, b);
  const hit = intersectLines(rayLine, edgeLine);
  if (!hit) return null;

  const t = Math.abs(dir[0]) > Math.abs(dir[1]) ? (hit[0] - origin[0]) / dir[0] : (hit[1] - origin[1]) / dir[1];
  const s = Math.abs(b[0] - a[0]) > Math.abs(b[1] - a[1]) ? (hit[0] - a[0]) / (b[0] - a[0]) : (hit[1] - a[1]) / (b[1] - a[1]);

  if (t < -1e-6 || s < -1e-6 || s > 1 + 1e-6) return null;
  return { point: hit, t };
}

export function triangleEdges(p1: Vec2, p2: Vec2, p3: Vec2): [Vec2, Vec2][] {
  return [
    [p1, p2],
    [p2, p3],
    [p3, p1],
  ];
}
