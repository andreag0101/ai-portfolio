import { Route, Routes } from "react-router-dom";
import Navbar from "./components/Navbar";
import RagChatWidget from "./components/RagChatWidget";
import Home from "./pages/Home";
import Segmentation from "./pages/Segmentation";
import Panorama from "./pages/Panorama";
import Disparity from "./pages/Disparity";
import Homography from "./pages/Homography";
import ProjectiveGeometry from "./pages/ProjectiveGeometry";
import CornerMatching from "./pages/CornerMatching";
import TextureClassification from "./pages/TextureClassification";
import FaceRecognition from "./pages/FaceRecognition";
import Calibration from "./pages/Calibration";
import CarDetection from "./pages/CarDetection";
import DeepLearning from "./pages/DeepLearning";
import DeepLearningProject from "./pages/DeepLearningProject";
import ResearchAssistant from "./pages/ResearchAssistant";

export default function App() {
  return (
    <div className="min-h-screen">
      <Navbar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/segmentation" element={<Segmentation />} />
        <Route path="/panorama" element={<Panorama />} />
        <Route path="/disparity" element={<Disparity />} />
        <Route path="/homography" element={<Homography />} />
        <Route path="/projective-geometry" element={<ProjectiveGeometry />} />
        <Route path="/corners" element={<CornerMatching />} />
        <Route path="/texture" element={<TextureClassification />} />
        <Route path="/face" element={<FaceRecognition />} />
        <Route path="/calibration" element={<Calibration />} />
        <Route path="/car-detection" element={<CarDetection />} />
        <Route path="/deep-learning" element={<DeepLearning />} />
        <Route path="/deep-learning/:slug" element={<DeepLearningProject />} />
        <Route path="/research-assistant" element={<ResearchAssistant />} />
      </Routes>
      <RagChatWidget />
    </div>
  );
}
