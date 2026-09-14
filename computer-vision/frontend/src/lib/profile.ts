export const profile = {
  name: "Andrea Goh",
  tagline: "AI/ML Engineer & Computer Vision Researcher",
  location: "Lafayette, IN",
  email: "andreagoh0101@gmail.com",
  linkedin: "https://linkedin.com/in/goh-andrea",
  bio: [
    "I currently work as an AI Development Intern at ArgonDigital, where I build RAG-based LLM applications and implement AI solutions across platforms. I recently graduated from Purdue University with a Master's degree in Computer Engineering, specializing in AI/ML, following an undergraduate degree in Aeronautical Engineering with a focus on Controls and Automation. As a Computer Vision Research Assistant at Purdue's Robotic Vision Lab, I develop multimodal transformer models and 3D reconstruction pipelines from first principles.",
    "I enjoy tackling challenging problems by breaking them down and building solutions from the ground up. The projects below reflect that approach, implementing classical computer vision algorithms such as homography, stereo geometry, and feature matching from their underlying mathematics rather than relying on high-level library calls. This has given me a deeper understanding of how these systems work and the ability to reason through and implement complex algorithms independently.",
  ],
};

export interface DegreeEntry {
  degree: string;
  detail: string;
  dates: string;
  bullets?: string[];
}

export interface EducationEntry {
  school: string;
  location: string;
  logo?: string;
  degrees: DegreeEntry[];
}

export const education: EducationEntry[] = [
  {
    school: "Purdue University",
    location: "West Lafayette, IN",
    logo: "/logos/purdue.jpg",
    degrees: [
      {
        degree: "M.S. Computer Engineering",
        detail: "Specialization: AI/ML · GPA 3.57/4.00",
        dates: "08/24 – 05/26",
      },
      {
        degree: "B.S. Aeronautical & Astronautical Engineering",
        detail: "Specialization: Controls · GPA 3.77/4.00",
        dates: "08/20 – 05/24",
        bullets: ["Minor in Mathematics", "Dean's List, 8/8 semesters"],
      },
    ],
  },
];

export interface ExperienceEntry {
  org: string;
  location: string;
  role: string;
  dates: string;
  logo?: string;
  bullets: string[];
  links?: { label: string; url: string }[];
}

export const experience: ExperienceEntry[] = [
  {
    org: "ArgonDigital",
    location: "Austin, TX · Remote",
    role: "AI Development Intern",
    dates: "07/26 – Present",
    logo: "/logos/argondigital.jpg",
    bullets: [
      "Building a RAG-based chat-with-AI feature on AWS Bedrock for grounded, context-aware document and project queries.",
      "Improved LLM-based requirements generation through prompt engineering and multi-model benchmarking on quality, consistency, latency, and cost.",
      "Designed and generalized an AI Output Scoring System — combining deterministic checks with LLM-based scoring — into a configurable tool usable across use cases.",
    ],
  },
  {
    org: "Purdue Robotic Vision Lab",
    location: "West Lafayette, IN",
    role: "Computer Vision Research Assistant",
    dates: "01/25 – 05/26",
    logo: "/logos/purdue.jpg",
    bullets: [
      "Built a custom tokenizer for a multimodal transformer that generates unseen sensor data across modalities, conditioned on sparse, incomplete inputs.",
      "Developed a 3D stereo reconstruction pipeline from scratch using classical vision theory and plain mathematics — no OpenCV or other CV libraries.",
      "Analyzed multimodal sensor data (LiDAR, RGB, IMU) to improve model robustness under real-world distribution shift for self-driving applications.",
    ],
  },
  {
    org: "Purdue University",
    location: "West Lafayette, IN",
    role: "Graduate Teaching Assistant",
    dates: "08/24 – 05/26",
    logo: "/logos/purdue.jpg",
    bullets: [
      "Head GTA for Advanced C Programming, leading a 20-member instructional team supporting 469 students.",
      "Head GTA for Python for Data Science, designing exams, grading scripts, and teaching ML concepts during office hours.",
      "GTA for Computer Security, mentoring students on secure coding and cryptographic implementations.",
    ],
  },
  {
    org: "Purdue TNT Lab · Samsung",
    location: "West Lafayette, IN",
    role: "Research Assistant",
    dates: "05/24 – 08/24",
    logo: "/logos/purdue.jpg",
    bullets: [
      "Co-authored a manuscript on ML-accelerated electrical impedance tomography (EIT) for damage detection in soft nanocomposite sensors.",
      "Automated a simulation-to-training pipeline generating 4,500+ FEM datasets, accelerating model training 3×.",
      "Developed an ML-accelerated EIT approach achieving 6× faster anomaly localization.",
    ],
    links: [
      {
        label: "Read the paper (engrXiv preprint)",
        url: "https://engrxiv.org/preprint/view/7867/version/10138",
      },
    ],
  },
  {
    org: "Thales",
    location: "Singapore",
    role: "Modelling and Simulations Engineer Intern",
    dates: "05/23 – 08/23",
    logo: "/logos/thales.jpg",
    bullets: [
      "Reduced COMSOL simulation runtime by 82% while maintaining over 99% accuracy through simulation and model optimization.",
      "Designed tunable piezoelectric lattice CAD models for sonar and hydrophone applications and delivered a functional prototype.",
    ],
  },
];
