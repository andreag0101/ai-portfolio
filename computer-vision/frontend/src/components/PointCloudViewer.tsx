import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

interface PointCloudViewerProps {
  positions: Float32Array; // (x, y, z) triples
  colors: Uint8Array; // (r, g, b) triples, 0-255
  pointSize?: number;
  height?: number;
}

/** An interactive (drag to rotate, scroll to zoom, right-drag to pan) WebGL
 * point cloud viewer, colored per-point from the source image. */
export default function PointCloudViewer({ positions, colors, pointSize = 2.2, height = 480 }: PointCloudViewerProps) {
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

    const material = new THREE.PointsMaterial({ size: pointSize, vertexColors: true });
    const points = new THREE.Points(geometry, material);
    scene.add(points);

    const sphere = geometry.boundingSphere ?? new THREE.Sphere(new THREE.Vector3(), 500);
    const center = sphere.center;
    const radius = Math.max(sphere.radius, 1);
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
      renderer.dispose();
      if (renderer.domElement.parentNode === container) {
        container.removeChild(renderer.domElement);
      }
    };
  }, [positions, colors, pointSize, height]);

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
