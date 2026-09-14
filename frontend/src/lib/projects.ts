export interface ProjectMeta {
  id: string;
  title: string;
  blurb: string;
  topics: string[];
  path?: string; // present if there's a live interactive demo
  thumbnail?: string;
}

export interface Category {
  id: string;
  title: string;
  description: string;
  projects: ProjectMeta[];
}

export const categories: Category[] = [
  {
    id: "geometry",
    title: "Geometric Vision",
    description: "Recovering 3D structure and undoing perspective distortion from 2D images.",
    projects: [
      {
        id: "projective-geometry",
        title: "Projective Geometry Playground",
        blurb: "Homogeneous points and lines, duality, and line intersection with cross products.",
        topics: ["Homogeneous coordinates", "Point-line duality"],
        path: "/projective-geometry",
        thumbnail: "/thumbnails/projective-geometry.jpg",
      },
      {
        id: "planar-rectification",
        title: "Planar Rectification & Compositing",
        blurb:
          "Automatically finds the dominant quadrilateral in a photo (a document, screen, sign, frame) and either straightens it to a fronto-parallel view, or warps a second image into it.",
        topics: ["Automatic quad detection", "DLT homography", "Perspective warping"],
        path: "/homography",
        thumbnail: "/thumbnails/planar-rectification.jpg",
      },
      {
        id: "camera-calibration",
        title: "Camera Calibration",
        blurb: "Zhang's method: from-scratch pattern corner detection, homographies, and closed-form intrinsics recovery.",
        topics: ["Zhang's method", "Intrinsics / extrinsics"],
        path: "/calibration",
        thumbnail: "/thumbnails/camera-calibration.jpg",
      },
    ],
  },
  {
    id: "features",
    title: "Features, Matching & Stitching",
    description: "Finding distinctive points across images and using them to relate or combine views.",
    projects: [
      {
        id: "corner-matching",
        title: "Corner Detection & Feature Matching",
        blurb: "From-scratch multiscale Harris corners matched across image pairs with SSD and NCC.",
        topics: ["Harris corners", "SSD / NCC matching"],
        path: "/corners",
        thumbnail: "/thumbnails/corner-matching.jpg",
      },
      {
        id: "panorama",
        title: "Panorama Stitcher",
        blurb: "SIFT correspondences, a custom RANSAC, and Levenberg-Marquardt refinement stitched into one image.",
        topics: ["SIFT", "RANSAC", "Nonlinear refinement"],
        path: "/panorama",
        thumbnail: "/thumbnails/panorama.jpg",
      },
    ],
  },
  {
    id: "segmentation",
    title: "Segmentation & Texture",
    description: "Separating an image into meaningful regions using color, intensity, and local texture.",
    projects: [
      {
        id: "image-segmentation",
        title: "Interactive Image Segmentation",
        blurb: "Iterative Otsu thresholding (color and texture) with morphological cleanup and contour extraction.",
        topics: ["Otsu thresholding", "Morphology"],
        path: "/segmentation",
        thumbnail: "/thumbnails/image-segmentation.jpg",
      },
      {
        id: "texture-classification",
        title: "Texture-Based Scene Classification",
        blurb: "Rotation-invariant Local Binary Patterns and deep Gram-matrix style features for weather classification.",
        topics: ["LBP", "Gram matrices"],
        path: "/texture",
        thumbnail: "/thumbnails/texture-classification.jpg",
      },
    ],
  },
  {
    id: "stereo",
    title: "Stereo & 3D",
    description: "Recovering depth and structure from two or more viewpoints of the same scene.",
    projects: [
      {
        id: "stereo-disparity",
        title: "Dense Stereo Depth Estimation",
        blurb: "Epipolar geometry and a census-transform windowed dense disparity map between a stereo pair.",
        topics: ["Epipolar geometry", "Dense stereo matching"],
        path: "/disparity",
        thumbnail: "/thumbnails/stereo-disparity.jpg",
      },
    ],
  },
  {
    id: "recognition",
    title: "Recognition & Detection",
    description: "Classical machine learning approaches to identifying faces and objects.",
    projects: [
      {
        id: "face-recognition",
        title: "Face Recognition: Eigenfaces vs. Fisherfaces",
        blurb: "PCA and Fisher-LDA subspace projections, nearest-neighbor matched against 30 people.",
        topics: ["PCA / eigenfaces", "Fisher-LDA"],
        path: "/face",
        thumbnail: "/thumbnails/face-recognition.jpg",
      },
      {
        id: "car-detection",
        title: "Car Detection: Haar Features + AdaBoost",
        blurb: "Integral-image Haar-like features boosted with AdaBoost, Viola-Jones style, classifying cropped patches.",
        topics: ["Haar features", "AdaBoost"],
        path: "/car-detection",
        thumbnail: "/thumbnails/car-detection.jpg",
      },
    ],
  },
];
