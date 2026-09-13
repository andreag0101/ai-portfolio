import { Route, Routes } from "react-router-dom";
import Navbar from "./components/Navbar";
import Home from "./pages/Home";
import Segmentation from "./pages/Segmentation";
import Panorama from "./pages/Panorama";
import Disparity from "./pages/Disparity";
import SparseReconstruction from "./pages/SparseReconstruction";
import Homography from "./pages/Homography";
import ProjectiveGeometry from "./pages/ProjectiveGeometry";
import CornerMatching from "./pages/CornerMatching";
import TextureClassification from "./pages/TextureClassification";
import FaceRecognition from "./pages/FaceRecognition";
import Calibration from "./pages/Calibration";
import CarDetection from "./pages/CarDetection";

export default function App() {
  return (
    <div className="min-h-screen">
      <Navbar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/segmentation" element={<Segmentation />} />
        <Route path="/panorama" element={<Panorama />} />
        <Route path="/disparity" element={<Disparity />} />
        <Route path="/sparse-reconstruction" element={<SparseReconstruction />} />
        <Route path="/homography" element={<Homography />} />
        <Route path="/projective-geometry" element={<ProjectiveGeometry />} />
        <Route path="/corners" element={<CornerMatching />} />
        <Route path="/texture" element={<TextureClassification />} />
        <Route path="/face" element={<FaceRecognition />} />
        <Route path="/calibration" element={<Calibration />} />
        <Route path="/car-detection" element={<CarDetection />} />
      </Routes>
    </div>
  );
}
