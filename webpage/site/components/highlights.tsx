"use client";

import { motion, useReducedMotion } from "framer-motion";
import { Zap, Layers, Globe } from "lucide-react";
import type { Dictionary } from "@/dictionaries/en";

interface HighlightsProps {
  dict: Dictionary["highlights"];
}

const icons = [Zap, Layers, Globe];

export function Highlights({ dict }: HighlightsProps) {
  const prefersReducedMotion = useReducedMotion();

  return (
    <section className="relative z-10 px-4 py-20 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-6xl">
        <motion.div
          initial={{ opacity: 0, y: prefersReducedMotion ? 0 : 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.35 }}
          transition={{ duration: 0.55, ease: "easeOut" }}
          className="mb-10 max-w-3xl"
        >
          <p className="ko-keep-all mb-4 text-xs font-semibold uppercase tracking-[0.2em] text-cyan-700/80">
            {dict.eyebrow}
          </p>
          <h2 className="ko-keep-all max-w-[18ch] text-3xl font-semibold tracking-[-0.04em] text-slate-950 sm:text-4xl lg:text-5xl">
            {dict.title}
          </h2>
        </motion.div>

        <div className="grid gap-4 md:grid-cols-3">
          {dict.items.map((item, index) => {
            const Icon = icons[index] ?? Globe;
            return (
              <motion.article
                key={item.title}
                initial={{
                  opacity: 0,
                  y: prefersReducedMotion ? 0 : 28,
                  scale: prefersReducedMotion ? 1 : 0.98,
                }}
                whileInView={{ opacity: 1, y: 0, scale: 1 }}
                viewport={{ once: true, amount: 0.3 }}
                transition={{
                  duration: 0.5,
                  delay: index * 0.08,
                  ease: "easeOut",
                }}
                whileHover={
                  prefersReducedMotion
                    ? undefined
                    : { y: -6, transition: { duration: 0.2 } }
                }
                className="rounded-[2rem] border border-slate-200/80 bg-white/88 p-7 shadow-[0_28px_80px_-34px_rgba(15,23,42,0.16)] backdrop-blur-xl"
              >
                <div className="mb-6 flex h-12 w-12 items-center justify-center rounded-2xl border border-slate-200 bg-cyan-50">
                  <Icon className="h-5 w-5 text-cyan-700" />
                </div>
                <p className="ko-keep-all mb-3 text-sm font-medium uppercase tracking-[0.18em] text-slate-500">
                  {item.meta}
                </p>
                <h3 className="ko-keep-all mb-3 max-w-[11ch] text-2xl font-semibold tracking-[-0.03em] text-slate-900">
                  {item.title}
                </h3>
                <p className="ko-keep-all text-base leading-relaxed text-slate-600">
                  {item.description}
                </p>
              </motion.article>
            );
          })}
        </div>
      </div>
    </section>
  );
}
