import { useMemo } from "react";
import PointCloudViewer from "./PointCloudViewer";
import { decodeBase64Float32, decodeBase64Uint8 } from "../lib/api";
import { featuredPointCloud as pointCloudData } from "../lib/featuredPointCloud";

/** A pre-computed point cloud baked into the frontend bundle, so the homepage
 * can render a live, draggable 3D preview with no API call / backend
 * dependency on load. */
export default function FeaturedPointCloud({ height = 260 }: { height?: number }) {
  const { positions, colors } = useMemo(
    () => ({
      positions: decodeBase64Float32(pointCloudData.positions),
      colors: decodeBase64Uint8(pointCloudData.colors),
    }),
    [],
  );

  return <PointCloudViewer positions={positions} colors={colors} height={height} />;
}
