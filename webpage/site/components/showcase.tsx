"use client";

import { useState } from "react";
import { AnimatePresence, motion, useReducedMotion, useScroll, useTransform } from "framer-motion";
import { MotionDiv } from "@/components/motion-wrapper";
import { useRef } from "react";
import type { Dictionary } from "@/dictionaries/en";

interface ShowcaseProps {
  dict: Dictionary["showcase"];
}

const screenshots = [
  "/screenshots/mlxcel-generate.webp",
  "/screenshots/mlxcel-server.webp",
  "/screenshots/mlxcel-vision.webp",
  "/screenshots/mlxcel-benchmark.webp",
  "/screenshots/mlxcel-models.webp",
];

export function Showcase({ dict }: ShowcaseProps) {
  const [active, setActive] = useState(0);
  const sectionRef = useRef<HTMLElement | null>(null);
  const prefersReducedMotion = useReducedMotion();
  const { scrollYProgress } = useScroll({
    target: sectionRef,
    offset: ["start end", "end start"],
  });
  const frameY = useTransform(
    scrollYProgress,
    [0, 1],
    [prefersReducedMotion ? 0 : 70, prefersReducedMotion ? 0 : -50]
  );
  const imageY = useTransform(
    scrollYProgress,
    [0, 1],
    [prefersReducedMotion ? 0 : -18, prefersReducedMotion ? 0 : 18]
  );
  const bgY = useTransform(
    scrollYProgress,
    [0, 1],
    [prefersReducedMotion ? 0 : 95, prefersReducedMotion ? 0 : -70]
  );

  return (
    <section ref={sectionRef} className="py-24 relative z-10 overflow-hidden">
      <motion.div
        aria-hidden="true"
        style={{ y: bgY }}
        className="absolute inset-x-0 top-20 mx-auto h-72 max-w-6xl rounded-full bg-gradient-to-r from-transparent via-cyan-300/12 to-amber-200/16 blur-3xl"
      />
      <div className="container mx-auto px-4">
        <MotionDiv
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-12"
        >
          <h2 className="text-3xl md:text-5xl font-semibold tracking-[-0.04em] mb-4">
            {dict.title}
          </h2>
          <p className="text-slate-600 max-w-2xl mx-auto text-lg leading-relaxed">
            {dict.subtitle}
          </p>
        </MotionDiv>

        <div className="flex justify-center gap-2 mb-8 flex-wrap">
          {dict.tabs.map((tab, i) => (
            <motion.button
              key={i}
              onClick={() => setActive(i)}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.35, delay: i * 0.06 }}
              whileHover={
                prefersReducedMotion
                  ? undefined
                  : { y: -2, transition: { duration: 0.18 } }
              }
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                active === i
                  ? "bg-brand-cyan/15 text-brand-cyan border border-brand-cyan/50 shadow-[0_8px_30px_-18px_rgba(34,211,238,0.8)]"
                  : "glass-panel text-slate-500 hover:text-slate-900 hover:bg-white/95"
              }`}
            >
              {tab.label}
            </motion.button>
          ))}
        </div>

        <MotionDiv
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          style={{ y: frameY }}
          className="max-w-5xl mx-auto"
        >
          <div className="overflow-hidden rounded-[1.75rem] border border-slate-200/90 bg-white shadow-[0_30px_90px_-36px_rgba(15,23,42,0.18)]">
            <AnimatePresence mode="wait">
              <MotionDiv
                key={active}
                initial={{
                  opacity: 0,
                  x: prefersReducedMotion ? 0 : 42,
                  scale: prefersReducedMotion ? 1 : 0.985,
                }}
                animate={{ opacity: 1, x: 0, scale: 1 }}
                exit={{
                  opacity: 0,
                  x: prefersReducedMotion ? 0 : -28,
                  scale: prefersReducedMotion ? 1 : 0.992,
                }}
                transition={{ duration: 0.3, ease: "easeOut" }}
              >
                <motion.img
                  src={screenshots[active]}
                  alt={dict.tabs[active].alt}
                  style={{ y: imageY }}
                  className="block h-auto w-full"
                />
              </MotionDiv>
            </AnimatePresence>
          </div>
          <motion.div
            key={`caption-${active}`}
            initial={{ opacity: 0, y: prefersReducedMotion ? 0 : 18 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, ease: "easeOut" }}
              className="mx-auto mt-8 max-w-3xl text-center"
          >
            <p className="mb-2 text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">
              {dict.tabs[active].title}
            </p>
            <p className="text-lg leading-relaxed text-slate-600">
              {dict.tabs[active].description}
            </p>
          </motion.div>
        </MotionDiv>
      </div>
    </section>
  );
}
