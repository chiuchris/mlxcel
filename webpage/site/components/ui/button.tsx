import { ButtonHTMLAttributes, forwardRef } from "react";
import { cn } from "@/lib/utils";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary" | "glass" | "ghost";
  size?: "sm" | "md" | "lg";
  asChild?: boolean;
}

const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = "primary", size = "md", ...props }, ref) => {
    const variants = {
      primary:
        "border border-cyan-500/20 bg-brand-cyan text-white font-bold shadow-[0_18px_40px_-22px_rgba(0,167,196,0.55)] hover:bg-cyan-600 hover:shadow-[0_22px_48px_-24px_rgba(0,167,196,0.45)]",
      secondary:
        "border border-slate-200 bg-white text-slate-700 hover:border-brand-purple/30 hover:bg-brand-purple/8 hover:text-brand-purple",
      glass:
        "glass-panel text-slate-800 hover:bg-white/95 hover:border-slate-200/90",
      ghost: "text-slate-500 hover:text-slate-900 hover:bg-slate-900/5",
    };

    const sizes = {
      sm: "px-3 py-1.5 text-sm",
      md: "px-6 py-3 text-base",
      lg: "px-8 py-4 text-lg",
    };

    return (
      <button
        ref={ref}
        className={cn(
          "relative inline-flex items-center justify-center gap-2 rounded-lg transition-all duration-300 active:scale-95 disabled:opacity-50 disabled:pointer-events-none cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400/40 focus-visible:ring-offset-2 focus-visible:ring-offset-white",
          variants[variant],
          sizes[size],
          className
        )}
        {...props}
      />
    );
  }
);
Button.displayName = "Button";

export { Button };
