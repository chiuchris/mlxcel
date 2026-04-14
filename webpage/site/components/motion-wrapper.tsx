"use client";

import { motion, HTMLMotionProps } from "framer-motion";

export function MotionDiv(props: HTMLMotionProps<"div">) {
  return <motion.div {...props} />;
}

export function MotionH1(props: HTMLMotionProps<"h1">) {
  return <motion.h1 {...props} />;
}

export function MotionP(props: HTMLMotionProps<"p">) {
  return <motion.p {...props} />;
}
