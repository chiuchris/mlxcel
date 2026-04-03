"use client";

import { SmartDownload } from "@/components/smart-download";
import { MotionDiv, MotionH1 } from "@/components/motion-wrapper";
import { motion, useReducedMotion, useScroll, useTransform } from "framer-motion";
import { useRef, useState } from "react";
import { Cpu, Zap, Terminal } from "lucide-react";
import type { Dictionary } from "@/dictionaries/en";
import type { ReleaseData } from "@/lib/release";

function generateStars() {
  return Array.from({ length: 18 }).map(() => ({
    x: Math.random() * 100,
    y: Math.random() * 100,
    s: Math.random() * 2 + 1,
    d: Math.random() * 5 + 2,
  }));
}

function Starfield() {
  const [stars] = useState(generateStars);

  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      {stars.map((star, i) => (
        <div
          key={i}
          className="absolute rounded-full bg-cyan-500/8 animate-pulse-slow"
          style={{
            left: `${star.x}%`,
            top: `${star.y}%`,
            width: `${star.s}px`,
            height: `${star.s}px`,
            animationDuration: `${star.d}s`,
            willChange: "opacity",
          }}
        />
      ))}
    </div>
  );
}

interface HeroProps {
  dict: Dictionary["hero"];
  release: ReleaseData | null;
}

export function Hero({ dict, release }: HeroProps) {
  const sectionRef = useRef<HTMLElement | null>(null);
  const prefersReducedMotion = useReducedMotion();
  const descriptionLines = dict.description.split("\n");
  const secondaryDescriptionLines = dict.description_secondary.split("\n");
  const { scrollYProgress } = useScroll({
    target: sectionRef,
    offset: ["start start", "end start"],
  });
  const gridY = useTransform(
    scrollYProgress,
    [0, 1],
    [0, prefersReducedMotion ? 0 : 80]
  );
  const starsY = useTransform(
    scrollYProgress,
    [0, 1],
    [0, prefersReducedMotion ? 0 : 140]
  );
  const contentY = useTransform(
    scrollYProgress,
    [0, 1],
    [0, prefersReducedMotion ? 0 : -36]
  );
  const glowLeftY = useTransform(
    scrollYProgress,
    [0, 1],
    [0, prefersReducedMotion ? 0 : 110]
  );
  const glowRightY = useTransform(
    scrollYProgress,
    [0, 1],
    [0, prefersReducedMotion ? 0 : 75]
  );
  const featureBadges = [
    { icon: Cpu, label: "Apple Silicon" },
    { icon: Zap, label: "Metal GPU" },
    { icon: Terminal, label: "Rust Native" },
  ];

  return (
    <section
      ref={sectionRef}
      className="relative flex min-h-screen flex-col items-center justify-center overflow-hidden bg-[linear-gradient(180deg,#fbfdff_0%,#f8fbfe_48%,#f2f7fb_100%)] px-4 pt-36 pb-18 sm:px-6 sm:pt-40 lg:px-8 lg:pt-44 xl:pt-32"
    >
      {/* Background Effects */}
      <div className="absolute inset-0 z-0">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,rgba(255,255,255,0.96),rgba(255,255,255,0.78)_42%,rgba(243,248,252,0.94)_100%)]" />
        <motion.div
          style={{ y: gridY }}
          className="absolute inset-0 bg-[radial-gradient(circle_at_top,rgba(255,255,255,0.72),rgba(255,255,255,0.2)_38%,transparent_64%)]"
        />
        <motion.div
          style={{ y: glowLeftY }}
          className="absolute left-[10%] top-24 h-56 w-56 rounded-full bg-cyan-200/14 blur-3xl"
        />
        <motion.div
          style={{ y: glowRightY }}
          className="absolute right-[10%] top-10 h-64 w-64 rounded-full bg-slate-200/18 blur-3xl"
        />
        <motion.div style={{ y: starsY }}>
          <Starfield />
        </motion.div>
      </div>

      <motion.div
        style={{ y: contentY }}
        className="container relative z-10 mx-auto flex max-w-6xl flex-col items-center gap-7 text-center"
      >
        <MotionDiv
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          className="mb-1 inline-flex items-center gap-2 rounded-full border border-cyan-200/70 bg-white/90 px-4 py-2 text-xs font-semibold uppercase tracking-[0.16em] text-cyan-700 shadow-[0_14px_32px_-24px_rgba(15,23,42,0.35)]"
        >
          {dict.badge}
        </MotionDiv>

        <MotionH1
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.8, delay: 0.1 }}
          className="max-w-5xl text-4xl font-semibold leading-[1.02] tracking-[-0.04em] text-slate-950 sm:text-5xl md:text-6xl lg:text-7xl xl:text-[5.25rem]"
        >
          <span className="block bg-clip-text pb-1 text-transparent bg-gradient-to-b from-slate-950 via-slate-900 to-slate-700 text-glow-white">
            {dict.title_prefix}
          </span>
          <span className="block pb-1 text-cyan-700 text-glow-cyan">
            {dict.title_suffix}
          </span>
          <span className="block text-slate-800">
            {dict.title_accent}
          </span>
        </MotionH1>

        <MotionDiv
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="max-w-[46rem] text-center"
        >
          <p className="ko-keep-all text-lg font-medium leading-[1.5] text-slate-700 sm:text-xl md:text-[1.55rem]">
            {descriptionLines.map((line, lineIndex) => (
              <span key={`${dict.description}-${lineIndex}`} className="block">
                {line}
              </span>
            ))}
          </p>
          <p className="ko-keep-all mx-auto mt-4 max-w-[38rem] text-sm leading-relaxed text-slate-500 sm:text-base md:text-lg">
            {secondaryDescriptionLines.map((line, lineIndex) => (
              <span
                key={`${dict.description_secondary}-${lineIndex}`}
                className="block"
              >
                {line}
              </span>
            ))}
          </p>
        </MotionDiv>

        <MotionDiv
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.25 }}
          className="flex flex-wrap items-center justify-center gap-3 text-sm text-slate-600"
        >
          {featureBadges.map(({ icon: Icon, label }) => (
            <motion.div
              key={label}
              initial={{ opacity: 0, y: 18, scale: 0.96 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              transition={{ duration: 0.45, delay: 0.35 }}
              className="flex items-center gap-2 rounded-full border border-slate-200 bg-white/92 px-3 py-2 shadow-[0_12px_30px_-24px_rgba(15,23,42,0.2)] backdrop-blur-md"
            >
              <Icon className="h-4 w-4 text-cyan-700" />
              <span>{label}</span>
            </motion.div>
          ))}
        </MotionDiv>

        <MotionDiv
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.3 }}
          className="mt-4"
        >
          <SmartDownload dict={dict} release={release} />
        </MotionDiv>
      </motion.div>
    </section>
  );
}
