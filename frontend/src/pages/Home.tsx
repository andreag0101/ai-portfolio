import { useState } from "react";
import { Link } from "react-router-dom";
import ProjectCard from "../components/ProjectCard";
import FlipCard from "../components/FlipCard";
import PanoramaStrip from "../components/PanoramaStrip";
import FeaturedPointCloud from "../components/FeaturedPointCloud";
import DLProjectCard from "../components/dl/DLProjectCard";
import { categories } from "../lib/projects";
import { dlCategories, dlProjects } from "../lib/deepLearningProjects";
import { profile, education, experience } from "../lib/profile";

const FEATURED_IDS = new Set(["stereo-disparity", "planar-rectification", "panorama"]);
const DL_FEATURED_IDS = new Set(["gan-diffusion", "object-detection"]);

function Avatar() {
  return (
    <img
      src="/profile/andrea.jpg"
      alt={profile.name}
      className="h-36 w-36 shrink-0 rounded-full object-cover object-[center_25%] ring-1 ring-neutral-200 dark:ring-neutral-800 sm:h-44 sm:w-44"
    />
  );
}

function SectionHeading({ eyebrow, title }: { eyebrow: string; title: string }) {
  return (
    <div className="mb-8">
      <p className="text-xs font-semibold uppercase tracking-wider text-gold-700 dark:text-gold-400">{eyebrow}</p>
      <h2 className="mt-1 text-2xl font-semibold tracking-tight text-neutral-900 dark:text-neutral-100">{title}</h2>
    </div>
  );
}

function LogoBadge({ src, alt }: { src: string; alt: string }) {
  return (
    <div className="flex h-10 max-w-[110px] shrink-0 items-center justify-center rounded-md bg-white px-2 py-1.5 ring-1 ring-neutral-200 dark:ring-neutral-700">
      <img src={src} alt={alt} className="h-full w-auto max-w-full object-contain" />
    </div>
  );
}

export default function Home() {
  return (
    <div>
      {/* About */}
      <section id="about" className="scroll-mt-16 border-b border-neutral-200 dark:border-neutral-800">
        <div className="mx-auto max-w-5xl px-6 py-16 sm:py-20">
          <div className="flex flex-col items-center gap-8 sm:flex-row sm:items-start">
            <Avatar />
            <div className="text-center sm:text-left">
              <h1 className="text-3xl font-semibold tracking-tight text-neutral-900 dark:text-neutral-100">{profile.name}</h1>
              <p className="mt-1 text-lg text-neutral-500 dark:text-neutral-400">{profile.tagline}</p>
              <div className="mt-4 space-y-3 text-neutral-600 dark:text-neutral-400">
                {profile.bio.map((p, i) => (
                  <p key={i}>{p}</p>
                ))}
              </div>
              <div className="mt-5 flex flex-wrap justify-center gap-4 text-sm sm:justify-start">
                <a href={`mailto:${profile.email}`} className="font-medium text-gold-700 hover:underline dark:text-gold-400">
                  {profile.email}
                </a>
                <a href={profile.linkedin} target="_blank" rel="noreferrer" className="font-medium text-gold-700 hover:underline dark:text-gold-400">
                  LinkedIn
                </a>
                <a href="/resume.pdf" download className="font-medium text-gold-700 hover:underline dark:text-gold-400">
                  Resume (PDF)
                </a>
                <span className="text-neutral-400 dark:text-neutral-600">{profile.location}</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Projects */}
      <section id="projects" className="scroll-mt-16 border-b border-neutral-200 dark:border-neutral-800">
        <div className="mx-auto max-w-5xl px-6 py-14">
          <SectionHeading eyebrow="Portfolio" title="Projects" />

          <div>
            <h3 className="text-xl font-semibold tracking-tight text-neutral-900 dark:text-neutral-100">
              Computer Vision, from scratch
            </h3>
            <p className="mt-3 max-w-2xl text-neutral-600 dark:text-neutral-400">
              A collection of classical computer vision techniques, implemented from first principles: no
              relying on OpenCV's built-in solvers for the core algorithms. Pick a card marked{" "}
              <span className="font-medium text-gold-700 dark:text-gold-400">Try it</span> to run it on
              your own images.
            </p>
            <p className="mt-3 max-w-2xl text-sm text-neutral-500 dark:text-neutral-500">
              Every project below was built using traditional computer vision methods and mathematics
              implemented by hand. OpenCV (or any other CV library) is used only where explicitly noted,
              e.g. as a baseline for comparison.
            </p>

            <FeaturedProjects />
            <AllProjects />
          </div>

          <div className="mt-16">
            <h3 className="text-xl font-semibold tracking-tight text-neutral-900 dark:text-neutral-100">
              Deep Learning
            </h3>
            <p className="mt-3 max-w-2xl text-neutral-600 dark:text-neutral-400">
              A collection of deep learning projects, from hand-derived backpropagation to generative
              models, implemented and trained in PyTorch: from scratch where the point is understanding
              the mechanics, and on top of established architectures where the point is the result.
            </p>
            <p className="mt-3 max-w-2xl text-sm text-neutral-500 dark:text-neutral-500">
              Each project below shows the trained results: figures, tables, and a full technical
              write-up. Nothing here runs live inference in your browser.
            </p>

            <DLFeaturedProjects />
            <DLAllProjects />
          </div>

          <div className="mt-16">
            <h3 className="text-xl font-semibold tracking-tight text-neutral-900 dark:text-neutral-100">
              AI Engineering
            </h3>
            <p className="mt-3 max-w-2xl text-neutral-600 dark:text-neutral-400">
              Retrieval and generation systems, built the same way as everything else here: from the
              ground up, and with the trade-offs made explicit.
            </p>

            <div className="mt-10 grid grid-cols-1 gap-6 lg:grid-cols-3">
              <FeaturedCard
                title="Research Assistant (RAG)"
                blurb="Ask questions across all 10 deep-learning write-ups on this site. From-scratch TF-IDF retrieval, Claude for grounded, cited generation."
                path="/research-assistant"
              >
                <RagPreview />
              </FeaturedCard>
            </div>
          </div>
        </div>
      </section>

      {/* Education */}
      <section id="education" className="scroll-mt-16 border-b border-neutral-200 dark:border-neutral-800">
        <div className="mx-auto max-w-5xl px-6 py-14">
          <SectionHeading eyebrow="Education" title="Where I studied" />
          <div className="space-y-6">
            {education.map((school) => (
              <div key={school.school} className="rounded-xl border border-neutral-200 bg-white p-5 dark:border-neutral-800 dark:bg-neutral-900">
                <div className="flex items-start justify-between gap-4">
                  <div className="flex items-center gap-3">
                    {school.logo && <LogoBadge src={school.logo} alt={school.school} />}
                    <h3 className="text-base font-semibold text-neutral-900 dark:text-neutral-100">{school.school}</h3>
                  </div>
                  <span className="shrink-0 text-sm text-neutral-500">{school.location}</span>
                </div>
                <div className="mt-4 space-y-4">
                  {school.degrees.map((d) => (
                    <div key={d.degree} className="border-l-2 border-neutral-200 pl-4 dark:border-neutral-800">
                      <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-0.5">
                        <p className="font-medium text-neutral-800 dark:text-neutral-200">{d.degree}</p>
                        <p className="text-sm text-neutral-500">{d.dates}</p>
                      </div>
                      <p className="text-sm text-neutral-500 dark:text-neutral-400">{d.detail}</p>
                      {d.bullets && (
                        <ul className="mt-1.5 list-disc space-y-0.5 pl-4 text-sm text-neutral-500 dark:text-neutral-400">
                          {d.bullets.map((b) => (
                            <li key={b}>{b}</li>
                          ))}
                        </ul>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Career */}
      <section id="career" className="scroll-mt-16">
        <div className="mx-auto max-w-5xl px-6 py-14">
          <SectionHeading eyebrow="Career" title="Where I've worked" />
          <div className="space-y-6">
            {experience.map((job) => (
              <div key={`${job.org}-${job.role}`} className="rounded-xl border border-neutral-200 bg-white p-5 dark:border-neutral-800 dark:bg-neutral-900">
                <div className="flex gap-3">
                  {job.logo && <LogoBadge src={job.logo} alt={job.org} />}
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-0.5">
                      <h3 className="text-base font-semibold text-neutral-900 dark:text-neutral-100">{job.role}</h3>
                      <span className="shrink-0 text-sm text-neutral-500">{job.dates}</span>
                    </div>
                    <p className="text-sm text-neutral-500 dark:text-neutral-400">
                      {job.org} · {job.location}
                    </p>
                  </div>
                </div>
                <ul className="mt-2.5 list-disc space-y-1 pl-4 text-sm text-neutral-600 dark:text-neutral-400">
                  {job.bullets.map((b) => (
                    <li key={b}>{b}</li>
                  ))}
                </ul>
                {job.links && (
                  <div className="mt-2.5 flex flex-wrap gap-4">
                    {job.links.map((l) => (
                      <a
                        key={l.url}
                        href={l.url}
                        target="_blank"
                        rel="noreferrer"
                        className="text-sm font-medium text-gold-700 hover:underline dark:text-gold-400"
                      >
                        {l.label} &rarr;
                      </a>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}

function FeaturedProjects() {
  return (
    <div className="mt-10 grid grid-cols-1 gap-6 lg:grid-cols-3">
      <FeaturedCard
        title="Dense Stereo Depth Estimation"
        blurb="Every pixel unprojected into a colored 3D point from a census-transform disparity map. Drag to rotate, scroll to zoom."
        path="/disparity"
      >
        <FeaturedPointCloud height={240} />
      </FeaturedCard>

      <FeaturedCard
        title="Planar Rectification"
        blurb="Automatic quadrilateral detection + a 4-point DLT homography straightens a photographed plane to fronto-parallel. Click to flip."
        path="/homography"
      >
        <FlipCard
          front="/featured/rectify-before.jpg"
          back="/featured/rectify-after.jpg"
          frontLabel="Before"
          backLabel="After"
          height={240}
        />
      </FeaturedCard>

      <FeaturedCard
        title="Panorama Stitcher"
        blurb="5 overlapping photos merged with SIFT, a from-scratch RANSAC, and Levenberg-Marquardt refinement."
        path="/panorama"
      >
        <PanoramaStrip src="/featured/panorama.jpg" height={240} />
      </FeaturedCard>
    </div>
  );
}

function FeaturedCard({
  title,
  blurb,
  path,
  cta = "Try it",
  children,
}: {
  title: string;
  blurb: string;
  path: string;
  cta?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-xl border border-neutral-200 bg-white p-4 dark:border-neutral-800 dark:bg-neutral-900">
      {children}
      <div className="mt-3 flex items-start justify-between gap-2">
        <div>
          <h3 className="text-base font-semibold text-neutral-900 dark:text-neutral-100">{title}</h3>
          <p className="mt-1 text-sm text-neutral-600 dark:text-neutral-400">{blurb}</p>
        </div>
      </div>
      <Link
        to={path}
        className="mt-3 inline-block rounded-md bg-neutral-900 px-3 py-1.5 text-sm font-medium text-white transition hover:bg-neutral-700 dark:bg-neutral-100 dark:text-neutral-900 dark:hover:bg-neutral-300"
      >
        {cta} &rarr;
      </Link>
    </div>
  );
}

function RagPreview() {
  return (
    <div className="h-[240px] overflow-hidden rounded-lg border border-neutral-200 dark:border-neutral-800">
      <img src="/thumbnails/research-assistant.jpg" alt="Research Assistant answering a question with a cited source" className="h-full w-full object-cover" />
    </div>
  );
}

function DLFeaturedProjects() {
  return (
    <div className="mt-10 grid grid-cols-1 gap-6 lg:grid-cols-2">
      <FeaturedCard
        title="GANs vs. Diffusion: Face Generation"
        blurb="A DCGAN trained from scratch on CelebA, benchmarked against a pretrained diffusion model by Fréchet Inception Distance. Click to flip."
        path="/deep-learning/gan-diffusion"
        cta="View project"
      >
        <FlipCard
          front="/deep-learning/gan-diffusion/gan-faces-wide.png"
          back="/deep-learning/gan-diffusion/diffusion-faces-wide.png"
          frontLabel="DCGAN"
          backLabel="Diffusion"
          height={240}
        />
      </FeaturedCard>

      <FeaturedCard
        title="Object Detection From Scratch on COCO"
        blurb="A single-shot detector's own predictions (green) against ground truth (red) for pizza, cat, and bus. Drag to pan."
        path="/deep-learning/object-detection"
        cta="View project"
      >
        <PanoramaStrip src="/deep-learning/object-detection/detections-strip.jpg" height={240} />
      </FeaturedCard>
    </div>
  );
}

function DLAllProjects() {
  const [show, setShow] = useState(false);

  const filteredCategories = dlCategories
    .map((cat) => ({ ...cat, projects: dlProjects.filter((p) => p.category === cat.id && !DL_FEATURED_IDS.has(p.slug)) }))
    .filter((cat) => cat.projects.length > 0);

  return (
    <div className="mt-10">
      <button
        onClick={() => setShow((s) => !s)}
        className="rounded-md bg-neutral-100 px-4 py-2 text-sm font-medium text-neutral-700 transition hover:bg-neutral-200 dark:bg-neutral-800 dark:text-neutral-300 dark:hover:bg-neutral-700"
      >
        {show ? "Hide remaining projects" : `See all projects (${dlProjects.length})`}
      </button>

      {show && (
        <div className="mt-8 space-y-12">
          {filteredCategories.map((cat) => (
            <div key={cat.id}>
              <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">{cat.title}</h3>
              <p className="mt-1 max-w-2xl text-sm text-neutral-500 dark:text-neutral-500">{cat.description}</p>
              <div className="mt-5 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
                {cat.projects.map((project) => (
                  <DLProjectCard key={project.slug} project={project} />
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function AllProjects() {
  const [show, setShow] = useState(false);

  const filteredCategories = categories
    .map((cat) => ({ ...cat, projects: cat.projects.filter((p) => !FEATURED_IDS.has(p.id)) }))
    .filter((cat) => cat.projects.length > 0);

  return (
    <div className="mt-10">
      <button
        onClick={() => setShow((s) => !s)}
        className="rounded-md bg-neutral-100 px-4 py-2 text-sm font-medium text-neutral-700 transition hover:bg-neutral-200 dark:bg-neutral-800 dark:text-neutral-300 dark:hover:bg-neutral-700"
      >
        {show ? "Hide remaining projects" : `See all projects (${categories.reduce((n, c) => n + c.projects.length, 0)})`}
      </button>

      {show && (
        <div className="mt-8 space-y-12">
          {filteredCategories.map((cat) => (
            <div key={cat.id}>
              <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">{cat.title}</h3>
              <p className="mt-1 max-w-2xl text-sm text-neutral-500 dark:text-neutral-500">{cat.description}</p>
              <div className="mt-5 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
                {cat.projects.map((project) => (
                  <ProjectCard key={project.id} project={project} />
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
