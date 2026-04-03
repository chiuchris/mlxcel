"use client";

import { motion, MotionProps } from "framer-motion";
import { ReactNode, HTMLAttributes } from "react";

// Explicitly defining the common motion props we use to avoid TS confusion
interface CustomMotionProps extends HTMLAttributes<HTMLDivElement> {
  children?: ReactNode;
  className?: string;
  initial?: any;
  animate?: any;
  exit?: any;
  whileHover?: any;
  whileTap?: any;
  whileInView?: any;
  transition?: any;
  viewport?: any;
  variants?: any;
  style?: any;
}

export function MotionDiv({ children, ...props }: CustomMotionProps) {
  return <motion.div {...(props as any)}>{children}</motion.div>;
}

export function MotionH1({
  children,
  ...props
}: CustomMotionProps & { as?: any }) {
  return <motion.h1 {...(props as any)}>{children}</motion.h1>;
}

export function MotionP({
  children,
  ...props
}: CustomMotionProps & { as?: any }) {
  return <motion.p {...(props as any)}>{children}</motion.p>;
}
