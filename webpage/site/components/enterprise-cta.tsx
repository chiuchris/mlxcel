"use client";

import { motion, useReducedMotion, useScroll, useTransform } from "framer-motion";
import { ArrowRight, Boxes, Building2, Cpu, Route } from "lucide-react";
import { MotionDiv } from "@/components/motion-wrapper";
import { useRef } from "react";
import type { Dictionary } from "@/dictionaries/en";

interface EnterpriseCtaProps {
  dict: Dictionary["enterprise"];
}

const pointIcons = [Cpu, Boxes, Route, Building2];

export function EnterpriseCta({ dict }: EnterpriseCtaProps) {
  const sectionRef = useRef<HTMLElement | null>(null);
  const prefersReducedMotion = useReducedMotion();
  const titleLines = dict.title.split("\n");
  const { scrollYProgress } = useScroll({
    target: sectionRef,
    offset: ["start end", "end start"],
  });
  const cardY = useTransform(
    scrollYProgress,
    [0, 1],
    [prefersReducedMotion ? 0 : 55, prefersReducedMotion ? 0 : -30]
  );
  const haloY = useTransform(
    scrollYProgress,
    [0, 1],
    [prefersReducedMotion ? 0 : 85, prefersReducedMotion ? 0 : -65]
  );

  return (
    <section ref={sectionRef} className="py-20 relative overflow-hidden">
      <motion.div
        aria-hidden="true"
        style={{ y: haloY }}
        className="absolute inset-x-0 top-16 mx-auto h-56 max-w-5xl rounded-full bg-gradient-to-r from-cyan-200/12 via-slate-200/10 to-amber-100/16 blur-3xl"
      />
      <div className="container mx-auto px-4">
        <MotionDiv
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          viewport={{ once: true }}
          style={{ y: cardY }}
          className="mx-auto max-w-5xl"
        >
          <motion.div
            initial={{
              opacity: 0,
              scale: prefersReducedMotion ? 1 : 0.96,
              y: prefersReducedMotion ? 0 : 24,
            }}
            whileInView={{ opacity: 1, scale: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.55, ease: "easeOut" }}
            whileHover={
              prefersReducedMotion
                ? undefined
                : { y: -4, transition: { duration: 0.2 } }
            }
            className="rounded-[2.25rem] border border-slate-200/80 bg-white/88 px-8 py-10 shadow-[0_28px_90px_-40px_rgba(15,23,42,0.18)] backdrop-blur-xl transition-colors hover:border-slate-300/90 md:px-12 md:py-12"
          >
            <div className="grid gap-8 lg:grid-cols-[1.05fr_0.95fr] lg:items-start lg:gap-12">
              <div className="min-w-0 text-left lg:pt-1">
                <p className="mb-4 text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">
                  {dict.badge}
                </p>
                <h3 className="max-w-[13ch] text-3xl font-semibold tracking-[-0.05em] text-slate-950 sm:text-4xl lg:text-[3.35rem] lg:leading-[1.02]">
                  {titleLines.map((line, lineIndex) => (
                    <span
                      key={`${dict.title}-${lineIndex}`}
                      className="block md:whitespace-nowrap"
                    >
                      {line}
                    </span>
                  ))}
                </h3>
                <a
                  href="mailto:contact@lablup.com"
                  className="mt-7 inline-flex items-center gap-2 rounded-full border border-slate-300 bg-white/80 px-5 py-3 text-sm font-medium text-slate-900 transition-all hover:border-slate-400 hover:bg-white"
                >
                  {dict.cta}
                  <ArrowRight className="w-4 h-4" />
                </a>
              </div>
              <div className="min-w-0 text-left">
                <p className="max-w-xl text-base leading-relaxed text-slate-600 keep-all sm:text-lg">
                  {dict.description}
                </p>
                <p className="mt-4 max-w-xl text-sm leading-relaxed text-slate-500 keep-all">
                  {dict.note}
                </p>
              </div>
            </div>

            <div className="mt-9 border-t border-slate-200/80 pt-6">
              <div className="mb-4 text-left text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-400">
                {dict.points_label}
              </div>
              <div className="grid gap-3 text-left sm:grid-cols-2 lg:grid-cols-4">
                {dict.points.map((point, index) => {
                  const Icon = pointIcons[index] ?? Boxes;

                  return (
                    <div
                      key={point.title}
                      className="flex min-h-[168px] flex-col rounded-[1.5rem] border border-slate-200/80 bg-slate-50/80 px-4 py-4"
                    >
                      <div className="mb-5 flex h-10 w-10 items-center justify-center rounded-2xl border border-slate-200 bg-white text-slate-700 shadow-[0_8px_24px_-18px_rgba(15,23,42,0.32)]">
                        <Icon className="h-4.5 w-4.5" />
                      </div>
                      <div className="space-y-2">
                        <p className="text-sm font-semibold leading-snug text-slate-900 keep-all">
                          {point.title}
                        </p>
                        <p className="text-xs leading-relaxed text-slate-500 keep-all">
                          {point.detail}
                        </p>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </motion.div>
        </MotionDiv>
      </div>
    </section>
  );
}
