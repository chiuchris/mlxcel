"use client";

import { motion, useReducedMotion } from "framer-motion";
import { Cpu, Terminal } from "lucide-react";
import type { Dictionary } from "@/dictionaries/en";

interface FeatureStoryProps {
  dict: Dictionary["story"];
}

const panelIcons = [Cpu, Terminal];
const panelAccents = [
  "from-cyan-300/18 via-cyan-200/6 to-transparent",
  "from-amber-200/16 via-purple-300/8 to-transparent",
];
const statAccents = [
  "border-cyan-100/90 bg-cyan-50/80",
  "border-amber-100/90 bg-orange-50/70",
];

export function FeatureStory({ dict }: FeatureStoryProps) {
  const prefersReducedMotion = useReducedMotion();

  return (
    <section className="relative z-10 overflow-hidden px-4 py-24 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-6xl">
        <motion.div
          initial={{ opacity: 0, y: prefersReducedMotion ? 0 : 26 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.35 }}
          transition={{ duration: 0.55, ease: "easeOut" }}
          className="mb-12 max-w-4xl"
        >
          <p className="ko-keep-all mb-4 text-xs font-semibold uppercase tracking-[0.2em] text-cyan-700/80">
            {dict.eyebrow}
          </p>
          <h2 className="feature-story__title ko-keep-all max-w-[15ch] text-3xl font-semibold tracking-[-0.04em] text-slate-950 sm:text-4xl lg:text-6xl">
            {dict.title}
          </h2>
          <p className="feature-story__subtitle ko-keep-all mt-5 max-w-3xl text-lg leading-relaxed text-slate-600">
            {dict.subtitle}
          </p>
        </motion.div>

        <div className="grid gap-5 lg:grid-cols-2">
          {dict.panels.map((panel, index) => {
            const Icon = panelIcons[index] ?? Terminal;
            const titleLines = panel.title.split("\n");
            return (
              <motion.article
                key={panel.title}
                initial={{
                  opacity: 0,
                  x: prefersReducedMotion ? 0 : index % 2 === 0 ? -32 : 32,
                  y: prefersReducedMotion ? 0 : 24,
                }}
                whileInView={{ opacity: 1, x: 0, y: 0 }}
                viewport={{ once: true, amount: 0.25 }}
                transition={{
                  duration: 0.6,
                  delay: index * 0.08,
                  ease: "easeOut",
                }}
                className="relative overflow-hidden rounded-[2.25rem] border border-slate-200/80 bg-white/88 p-8 shadow-[0_30px_90px_-38px_rgba(15,23,42,0.18)] backdrop-blur-xl sm:p-10"
              >
                <div
                  className={`absolute inset-0 bg-gradient-to-br ${panelAccents[index] ?? panelAccents[0]}`}
                />
                <div className="relative z-10">
                  <div className="mb-8 flex items-start justify-between gap-6">
                    <div className="min-w-0 flex-1">
                      <p className="ko-keep-all mb-3 text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">
                        {panel.eyebrow}
                      </p>
                      <h3 className="feature-story__panel-title ko-keep-all max-w-[12ch] text-3xl font-semibold leading-[1.05] tracking-[-0.04em] text-slate-900 sm:text-4xl">
                        {titleLines.map((line, lineIndex) => (
                          <span
                            key={`${panel.title}-${lineIndex}`}
                            className="feature-story__panel-title-line block"
                          >
                            {line}
                          </span>
                        ))}
                      </h3>
                    </div>
                    <div className="flex h-12 w-12 items-center justify-center rounded-2xl border border-slate-200 bg-white/82">
                      <Icon className="h-5 w-5 text-cyan-700" />
                    </div>
                  </div>

                  <p className="ko-keep-all max-w-xl text-base leading-relaxed text-slate-600 sm:text-lg">
                    {panel.description}
                  </p>

                  <div className="mt-8 grid gap-3">
                    {panel.points.map((point) => (
                      <div
                        key={point}
                        className="ko-keep-all rounded-2xl border border-slate-200/80 bg-white/72 px-4 py-4 text-sm leading-relaxed text-slate-700"
                      >
                        {point}
                      </div>
                    ))}
                  </div>

                  <div
                    className={`mt-10 inline-flex min-w-[220px] flex-col rounded-[1.75rem] border px-5 py-4 shadow-[0_18px_40px_-34px_rgba(15,23,42,0.16)] ${statAccents[index] ?? statAccents[0]}`}
                  >
                    <div className="flex items-center gap-2 text-slate-500">
                      <span className="flex h-6 w-6 items-center justify-center rounded-full bg-white/75">
                        <Icon className="h-3.5 w-3.5 text-cyan-700" />
                      </span>
                      <span className="ko-keep-all text-xs font-medium">
                        {panel.stat_label}
                      </span>
                    </div>
                    <span className="ko-keep-all mt-2 text-3xl font-semibold tracking-[-0.04em] text-slate-950">
                      {panel.stat_value}
                    </span>
                  </div>
                </div>
              </motion.article>
            );
          })}
        </div>
      </div>
    </section>
  );
}
