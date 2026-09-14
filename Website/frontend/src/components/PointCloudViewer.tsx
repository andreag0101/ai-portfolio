import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

interface PointCloudViewerProps {
  positions: Float32Array; // (x, y, z) triples
  colors: Uint8Array; // (r, g, b) triples, 0-255
  /** World-space splat diameter. Omit to auto-size from point spacing (bounding
   * sphere radius / sqrt(point count)) so neighboring splats overlap and the
   * surface looks filled in rather than dotted, regardless of how sparse the
   * underlying point count is. */
  pointSize?: number;
  /** Multiplier on the auto-computed splat size above. Only applies when
   * `pointSize` is omitted. */
  splatScale?: number;
  height?: number;
}

/** Draws a soft, radially-fading white disc into a canvas texture -- point-
 * splatting (Pfister/Zwicker-style point-based rendering, not the 2023
 * per-scene-optimized "3D Gaussian Splatting"), so each point in a sparse
 * cloud renders as a soft Gaussian-falloff blob that blends into its
 * neighbors instead of a tiny hard dot with visible gaps between points. */
function createSplatTexture(): THREE.Texture {
  const size = 64;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d")!;
  const grad = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  grad.addColorStop(0.0, "rgba(255,255,255,1)");
  grad.addColorStop(0.4, "rgba(255,255,255,0.8)");
  grad.addColorStop(0.75, "rgba(255,255,255,0.25)");
  grad.addColorStop(1.0, "rgba(255,255,255,0)");
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, size, size);
  const texture = new THREE.CanvasTexture(canvas);
  texture.needsUpdate = true;
  return texture;
}

/** An interactive (drag to rotate, scroll to zoom, right-drag to pan) WebGL
 * point cloud viewer, colored per-point from the source image. */
export default function PointCloudViewer({ positions, colors, pointSize, splatScale = 4.5, height = 480 }: PointCloudViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container || positions.length === 0) return;

    const width = container.clientWidth;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0c0c10);

    const camera = new THREE.PerspectiveCamera(55, width / height, 1, 100000);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    const colorsF32 = new Float32Array(colors.length);
    for (let i = 0; i < colors.length; i++) colorsF32[i] = colors[i] / 255;
    geometry.setAttribute("color", new THREE.BufferAttribute(colorsF32, 3));
    geometry.computeBoundingSphere();

    const sphere = geometry.boundingSphere ?? new THREE.Sphere(new THREE.Vector3(), 500);
    const center = sphere.center;
    const radius = Math.max(sphere.radius, 1);

    const numPoints = positions.length / 3;
    // average spacing between points, assuming they're roughly spread over a
    // 2D surface within the bounding sphere: radius / sqrt(count)
    const avgSpacing = radius / Math.sqrt(numPoints);
    const autoSize = avgSpacing * splatScale;
    const size = pointSize ?? Math.min(Math.max(autoSize, radius * 0.01), radius * 0.15);

    const splatTexture = createSplatTexture();
    const material = new THREE.PointsMaterial({
      size,
      vertexColors: true,
      map: splatTexture,
      transparent: true,
      alphaTest: 0.02,
      depthWrite: false,
      sizeAttenuation: true,
    });
    const points = new THREE.Points(geometry, material);
    scene.add(points);

    camera.position.set(center.x, center.y, center.z + radius * 1.8);
    camera.near = radius * 0.01;
    camera.far = radius * 30;
    camera.updateProjectionMatrix();

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.copy(center);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = radius * 0.05;
    controls.maxDistance = radius * 15;
    controls.update();

    let raf = 0;
    function animate() {
      controls.update();
      renderer.render(scene, camera);
      raf = requestAnimationFrame(animate);
    }
    animate();

    function handleResize() {
      if (!container) return;
      const w = container.clientWidth;
      camera.aspect = w / height;
      camera.updateProjectionMatrix();
      renderer.setSize(w, height);
    }
    window.addEventListener("resize", handleResize);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", handleResize);
      controls.dispose();
      geometry.dispose();
      material.dispose();
      splatTexture.dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode === container) {
        container.removeChild(renderer.domElement);
      }
    };
  }, [positions, colors, pointSize, splatScale, height]);

  if (positions.length === 0) {
    return (
      <div
        style={{ height }}
        className="flex items-center justify-center rounded-lg border border-dashed border-neutral-300 text-sm text-neutral-400 dark:border-neutral-700"
      >
        No valid depth points to reconstruct
      </div>
    );
  }

  return <div ref={containerRef} style={{ height }} className="w-full overflow-hidden rounded-lg border border-neutral-200 dark:border-neutral-800" />;
}
