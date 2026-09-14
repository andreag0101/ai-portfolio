// In dev, "/api" is proxied to localhost:8000 (see vite.config.ts). In
// production the frontend and backend are deployed separately, so this
// points at the deployed backend's URL via a build-time env var (set
// VITE_API_BASE, e.g. "https://<render-service>.onrender.com/api", in the
// hosting provider's project settings).
const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "/api";

async function unwrap<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (body?.detail) detail = body.detail;
    } catch {
      // ignore non-JSON error bodies
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export async function dataUrlToFile(dataUrl: string, name: string): Promise<File> {
  const res = await fetch(dataUrl);
  const blob = await res.blob();
  return new File([blob], name, { type: blob.type });
}

export interface SegmentationResult {
  input: string;
  mask: string;
  maskBeforeIteration: string;
  contour: string;
  openingMask: string;
  openingContour: string;
  closingMask: string;
  closingContour: string;
}

export async function runSegmentation(
  file: File,
  opts: { mode: "rgb" | "texture"; bins: number; flip: boolean; iterations: number },
): Promise<SegmentationResult> {
  const form = new FormData();
  form.append("file", file);
  form.append("mode", opts.mode);
  form.append("bins", String(opts.bins));
  form.append("flip", String(opts.flip));
  form.append("iterations", String(opts.iterations));
  const res = await fetch(`${API_BASE}/segmentation`, { method: "POST", body: form });
  return unwrap<SegmentationResult>(res);
}

export async function getSegmentationSamples(): Promise<{ samples: string[] }> {
  const res = await fetch(`${API_BASE}/segmentation/samples`);
  return unwrap<{ samples: string[] }>(res);
}

export interface PanoramaResult {
  keypointOverlays: string[];
  matchOverlays: string[];
  panorama: string;
}

export async function runPanorama(files: File[], const_: number): Promise<PanoramaResult> {
  const form = new FormData();
  for (const f of files) form.append("files", f);
  form.append("const", String(const_));
  const res = await fetch(`${API_BASE}/panorama`, { method: "POST", body: form });
  return unwrap<PanoramaResult>(res);
}

export async function getPanoramaSample(): Promise<{ images: string[] }> {
  const res = await fetch(`${API_BASE}/panorama/sample`);
  return unwrap<{ images: string[] }>(res);
}

export interface PointCloudData {
  positions: string; // base64-encoded Float32Array, (x,y,z) triples
  colors: string; // base64-encoded Uint8Array, (r,g,b) triples
  numPoints: number;
}

export function decodeBase64Float32(b64: string): Float32Array {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

export function decodeBase64Uint8(b64: string): Uint8Array {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

export interface DisparityResult {
  left: string;
  right: string;
  disparityGray: string;
  disparity: string;
  filteredDisparity: string;
  filteredKeepFrac: number;
  sgbmDisparity: string;
  sgbmKeepFrac: number;
  pointCloud: PointCloudData;
  sgbmPointCloud: PointCloudData;
}

export async function getDisparitySample(): Promise<{ left: string; right: string }> {
  const res = await fetch(`${API_BASE}/disparity/sample`);
  return unwrap<{ left: string; right: string }>(res);
}

export type Quad = [number, number][];

export interface QuadCandidate {
  quad: Quad;
  overlay: string; // data URL with the candidate outlined on the source photo
}

export interface CandidatesResult {
  image: string;
  candidates: QuadCandidate[];
}

export async function findQuadCandidates(file: File): Promise<CandidatesResult> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/rectify/candidates`, { method: "POST", body: form });
  return unwrap<CandidatesResult>(res);
}

export interface DewarpResult {
  input: string;
  detectedQuad: string;
  rectified: string;
}

export async function runDewarp(file: File, quad?: Quad): Promise<DewarpResult> {
  const form = new FormData();
  form.append("file", file);
  if (quad) form.append("quad", JSON.stringify(quad));
  const res = await fetch(`${API_BASE}/rectify/dewarp`, { method: "POST", body: form });
  return unwrap<DewarpResult>(res);
}

export interface InsertResult {
  dest: string;
  detectedQuad: string;
  composited: string;
}

export async function runInsert(dest: File, source: File, quad?: Quad): Promise<InsertResult> {
  const form = new FormData();
  form.append("dest", dest);
  form.append("source", source);
  if (quad) form.append("quad", JSON.stringify(quad));
  const res = await fetch(`${API_BASE}/rectify/insert`, { method: "POST", body: form });
  return unwrap<InsertResult>(res);
}

export async function getRectifySample(): Promise<{ image: string }> {
  const res = await fetch(`${API_BASE}/rectify/sample`);
  return unwrap<{ image: string }>(res);
}

export async function getRectifyInsertSample(): Promise<{ dest: string; source: string }> {
  const res = await fetch(`${API_BASE}/rectify/insert-sample`);
  return unwrap<{ dest: string; source: string }>(res);
}

export interface CornerMatchResult {
  cornersOverlay1: string;
  cornersOverlay2: string;
  combined: string;
  orbCombined: string;
  numCorners1: number;
  numCorners2: number;
  numMatches: number;
}

export async function runCornerMatch(
  file1: File,
  file2: File,
  opts: { sigma: number; distType: "SSD" | "NCC" },
): Promise<CornerMatchResult> {
  const form = new FormData();
  form.append("file1", file1);
  form.append("file2", file2);
  form.append("sigma", String(opts.sigma));
  form.append("distType", opts.distType);
  const res = await fetch(`${API_BASE}/corners/match`, { method: "POST", body: form });
  return unwrap<CornerMatchResult>(res);
}

export async function getCornersSample(): Promise<{ file1: string; file2: string }> {
  const res = await fetch(`${API_BASE}/corners/sample`);
  return unwrap<{ file1: string; file2: string }>(res);
}

export interface TextureNeighbor {
  label: string;
  distance: number;
  thumbnail: string;
}

export interface TextureResult {
  predicted: string;
  confidences: Record<string, number>;
  nearest: TextureNeighbor[];
  resizedInput: string;
  hueVisual: string;
  encodingVisual: string;
  histogram: number[];
}

export async function runTextureClassify(file: File): Promise<TextureResult> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/texture/classify`, { method: "POST", body: form });
  return unwrap<TextureResult>(res);
}

export async function getTextureSamples(): Promise<{ samples: string[] }> {
  const res = await fetch(`${API_BASE}/texture/samples`);
  return unwrap<{ samples: string[] }>(res);
}

export interface FaceMatch {
  label: number;
  distance: number;
  thumbnail: string;
}

export interface FaceClassifyResult {
  pca: FaceMatch;
  lda: FaceMatch;
  resizedInput: string;
  eigenfaces: string[];
  fisherfaces: string[];
}

export async function runFaceClassify(file: File, k: number): Promise<FaceClassifyResult> {
  const form = new FormData();
  form.append("file", file);
  form.append("k", String(k));
  const res = await fetch(`${API_BASE}/face/classify`, { method: "POST", body: form });
  return unwrap<FaceClassifyResult>(res);
}

export interface FaceAccuracyCurve {
  k: number[];
  pca: number[];
  lda: number[];
}

export async function getFaceAccuracyCurve(): Promise<FaceAccuracyCurve> {
  const res = await fetch(`${API_BASE}/face/accuracy-curve`);
  return unwrap<FaceAccuracyCurve>(res);
}

export async function getFaceSamples(): Promise<{ samples: string[] }> {
  const res = await fetch(`${API_BASE}/face/samples`);
  return unwrap<{ samples: string[] }>(res);
}

export interface CalibrationResult {
  cornerOverlays: string[];
  labeledOverlays: string[];
  reprojBefore: string[];
  reprojAfter: string[];
  intrinsics: { fx: number; fy: number; cx: number; cy: number; skew: number };
  posePlot: string;
}

export async function runCalibration(files: File[]): Promise<CalibrationResult> {
  const form = new FormData();
  for (const f of files) form.append("files", f);
  const res = await fetch(`${API_BASE}/calibration`, { method: "POST", body: form });
  return unwrap<CalibrationResult>(res);
}

export async function runCalibrationSample(): Promise<CalibrationResult> {
  const res = await fetch(`${API_BASE}/calibration/run-sample`, { method: "POST" });
  return unwrap<CalibrationResult>(res);
}

export async function getCalibrationPattern(): Promise<{ pattern: string }> {
  const res = await fetch(`${API_BASE}/calibration/pattern`);
  return unwrap<{ pattern: string }>(res);
}

export async function getCalibrationSamples(): Promise<{ samples: string[] }> {
  const res = await fetch(`${API_BASE}/calibration/samples`);
  return unwrap<{ samples: string[] }>(res);
}

export interface CarDetectionResult {
  isCar: boolean;
  score: number;
  threshold: number;
  numRounds: number;
  testAccuracy: number;
  confusion: { tp: number; fp: number; fn: number; tn: number };
  resizedPatch: string;
  featureOverlay: string;
}

export async function runCarDetection(file: File): Promise<CarDetectionResult> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/car-detection/classify`, { method: "POST", body: form });
  return unwrap<CarDetectionResult>(res);
}

export async function getCarDetectionSamples(): Promise<{ positive: string[]; negative: string[] }> {
  const res = await fetch(`${API_BASE}/car-detection/samples`);
  return unwrap<{ positive: string[]; negative: string[] }>(res);
}

export async function runDisparity(
  left: File,
  right: File,
  opts: { window: number; dMax: number },
): Promise<DisparityResult> {
  const form = new FormData();
  form.append("left", left);
  form.append("right", right);
  form.append("window", String(opts.window));
  form.append("d_max", String(opts.dMax));
  const res = await fetch(`${API_BASE}/disparity`, { method: "POST", body: form });
  return unwrap<DisparityResult>(res);
}

