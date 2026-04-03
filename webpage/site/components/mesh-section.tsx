"use client";

import { motion, useReducedMotion } from "framer-motion";
import { Layers, Building2 } from "lucide-react";
import type { Dictionary } from "@/dictionaries/en";

interface MeshSectionProps {
  dict: Dictionary["mesh"];
}

export function MeshSection({ dict }: MeshSectionProps) {
  const prefersReducedMotion = useReducedMotion();
  const titleLines = dict.title.split("\n");
  const meshCardTitleLines = dict.mesh_card.title.split("\n");
  const integrationTitleLines = dict.integration_card.title.split("\n");

  return (
    <section className="relative z-10 px-4 py-24 sm:px-6 lg:px-8">
      <div className="mx-auto grid max-w-6xl gap-6 lg:grid-cols-[1.2fr_0.8fr]">
        <motion.div
          initial={{ opacity: 0, y: prefersReducedMotion ? 0 : 28 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.25 }}
          transition={{ duration: 0.6, ease: "easeOut" }}
          className="relative px-2 py-2 sm:px-4 sm:py-4 lg:px-6 lg:py-6"
        >
          <div className="relative z-10">
            <p className="ko-keep-all mb-4 text-xs font-semibold uppercase tracking-[0.22em] text-cyan-700/80">
              {dict.eyebrow}
            </p>
            <h2 className="mesh-section__title ko-keep-all max-w-[15ch] text-3xl font-semibold tracking-[-0.05em] text-slate-950 sm:text-4xl lg:text-6xl">
              {titleLines.map((line, lineIndex) => (
                <span
                  key={`${dict.title}-${lineIndex}`}
                  className="mesh-section__title-line block"
                >
                  {line}
                </span>
              ))}
            </h2>
            <p className="ko-keep-all mt-5 max-w-2xl text-lg leading-relaxed text-slate-600">
              {dict.subtitle}
            </p>

            <div className="mt-10 grid gap-4 md:grid-cols-3">
              {dict.mesh_points.map((point, index) => (
                <motion.div
                  key={point.title}
                  initial={{
                    opacity: 0,
                    y: prefersReducedMotion ? 0 : 24,
                    scale: prefersReducedMotion ? 1 : 0.98,
                  }}
                  whileInView={{ opacity: 1, y: 0, scale: 1 }}
                  viewport={{ once: true, amount: 0.35 }}
                  transition={{
                    duration: 0.45,
                    delay: index * 0.08,
                    ease: "easeOut",
                  }}
                  className="flex h-full flex-col rounded-[1.75rem] border border-slate-200/80 bg-white/86 p-5 backdrop-blur-md"
                >
                  <p className="ko-keep-all mb-3 text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">
                    {point.step}
                  </p>
                  <h3 className="mesh-section__step-title ko-keep-all mb-2 max-w-[12ch] text-xl font-semibold leading-[1.15] tracking-[-0.03em] text-slate-900 md:min-h-[3.1rem]">
                    {point.title.split("\n").map((line, lineIndex) => (
                      <span
                        key={`${point.title}-${lineIndex}`}
                        className="mesh-section__step-title-line block"
                      >
                        {line}
                      </span>
                    ))}
                  </h3>
                  <p className="ko-keep-all flex-1 text-sm leading-relaxed text-slate-600">
                    {point.description}
                  </p>
                  {dict.mesh_badges[index] && (
                    <span className="ko-keep-all mt-6 inline-flex w-fit rounded-full border border-slate-200 bg-white px-4 py-2 text-sm text-slate-600 shadow-[0_12px_30px_-24px_rgba(15,23,42,0.14)]">
                      {dict.mesh_badges[index]}
                    </span>
                  )}
                </motion.div>
              ))}
            </div>
          </div>
        </motion.div>

        <motion.aside
          initial={{ opacity: 0, y: prefersReducedMotion ? 0 : 28 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.25 }}
          transition={{ duration: 0.6, delay: 0.08, ease: "easeOut" }}
          className="grid gap-6"
        >
          <article className="rounded-[2rem] border border-slate-200/80 bg-white/92 p-7 shadow-[0_24px_80px_-30px_rgba(15,23,42,0.16)] backdrop-blur-xl">
            <div className="mb-6 flex h-12 w-12 items-center justify-center rounded-2xl border border-slate-200 bg-cyan-50">
              <Layers className="h-5 w-5 text-cyan-700" />
            </div>
            <p className="ko-keep-all mb-3 text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">
              {dict.mesh_card.eyebrow}
            </p>
            <h3 className="mesh-section__card-title ko-keep-all mb-4 max-w-[12ch] text-2xl font-semibold leading-[1.15] tracking-[-0.03em] text-slate-900">
              {meshCardTitleLines.map((line, lineIndex) => (
                <span
                  key={`${dict.mesh_card.title}-${lineIndex}`}
                  className="mesh-section__card-title-line block"
                >
                  {line}
                </span>
              ))}
            </h3>
            <p className="ko-keep-all text-base leading-relaxed text-slate-600">
              {dict.mesh_card.description}
            </p>
          </article>

          <article className="rounded-[2rem] border border-amber-100 bg-amber-50/80 p-7 shadow-[0_24px_80px_-30px_rgba(15,23,42,0.12)]">
            <div className="mb-6 flex h-12 w-12 items-center justify-center rounded-2xl border border-amber-100 bg-white/80">
              <Building2 className="h-5 w-5 text-amber-600" />
            </div>
            <p className="ko-keep-all mb-3 text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">
              {dict.integration_card.eyebrow}
            </p>
            <h3 className="mesh-section__card-title ko-keep-all mb-4 max-w-[12ch] text-2xl font-semibold leading-[1.15] tracking-[-0.03em] text-slate-900">
              {integrationTitleLines.map((line, lineIndex) => (
                <span
                  key={`${dict.integration_card.title}-${lineIndex}`}
                  className="mesh-section__card-title-line block"
                >
                  {line}
                </span>
              ))}
            </h3>
            <p className="ko-keep-all text-base leading-relaxed text-slate-600">
              {dict.integration_card.description}
            </p>
          </article>
        </motion.aside>
      </div>
    </section>
  );
}
