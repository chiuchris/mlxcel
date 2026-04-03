"use client";

import { useState } from "react";
import { usePathname, useRouter } from "next/navigation";
import { Globe, Book, Menu, X, ArrowUpRight } from "lucide-react";
import type { Dictionary } from "@/dictionaries/en";

interface LanguageSwitcherProps {
  dict: Dictionary["announcement"];
}

export function LanguageSwitcher({ dict }: LanguageSwitcherProps) {
  const pathname = usePathname();
  const router = useRouter();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const currentLang = pathname.split("/")[1] || "en";

  const toggleLanguage = () => {
    const newLang = currentLang === "ko" ? "en" : "ko";
    const newPath = pathname.replace(`/${currentLang}`, `/${newLang}`);
    router.push(newPath);
    setMobileMenuOpen(false);
  };

  const buttonClass =
    "flex items-center gap-2 rounded-full border border-white/45 bg-white/58 px-3 py-2 text-sm text-slate-600 shadow-[0_12px_30px_-22px_rgba(15,23,42,0.2)] backdrop-blur-xl transition-colors hover:bg-white/72 hover:text-slate-900 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400/40 focus-visible:ring-offset-2 focus-visible:ring-offset-[#f7fbfe]";
  const mobileButtonClass =
    "flex items-center gap-2 rounded-full border border-white/45 bg-white/58 px-3 py-2 text-sm text-slate-700 shadow-[0_12px_30px_-22px_rgba(15,23,42,0.2)] backdrop-blur-xl transition-colors hover:bg-white/72 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400/40 focus-visible:ring-offset-2 focus-visible:ring-offset-[#f7fbfe]";
  const announcementClass =
    "fixed left-1/2 top-4 z-50 hidden -translate-x-1/2 items-center gap-2 rounded-full border border-white/45 bg-white/62 px-2.5 py-2 shadow-[0_18px_36px_-26px_rgba(15,23,42,0.24)] backdrop-blur-xl transition-colors hover:bg-white/78 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400/40 focus-visible:ring-offset-2 focus-visible:ring-offset-[#f7fbfe] xl:flex";
  const mobileAnnouncementClass =
    "fixed inset-x-4 top-[4.85rem] z-40 flex items-center justify-between gap-3 rounded-full border border-white/45 bg-white/72 px-3 py-2 shadow-[0_18px_36px_-26px_rgba(15,23,42,0.24)] backdrop-blur-xl transition-colors hover:bg-white/80 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400/40 focus-visible:ring-offset-2 focus-visible:ring-offset-[#f7fbfe] lg:hidden";

  return (
    <>
      {/* Logo */}
      <a
        href="https://mlxcel.ai"
        className="fixed top-4 left-4 z-50 inline-flex items-center rounded-full bg-white/58 px-4 py-2.5 shadow-[0_12px_30px_-22px_rgba(15,23,42,0.2)] backdrop-blur-xl transition-colors hover:bg-white/72 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400/40 focus-visible:ring-offset-2 focus-visible:ring-offset-[#f7fbfe]"
      >
        <span className="font-mono text-sm font-bold tracking-tight text-slate-900">mlxcel</span>
      </a>

      {dict.visible && (
        <a
          href={dict.href}
          target="_blank"
          rel="noopener noreferrer"
          className={announcementClass}
          aria-label={`${dict.text} (${dict.date})`}
        >
          <span className="rounded-full bg-[linear-gradient(135deg,rgba(251,191,36,0.18),rgba(251,146,60,0.12))] px-2.5 py-1 text-[0.64rem] font-semibold uppercase tracking-[0.18em] text-amber-700">
            {dict.tag}
          </span>
          <span className="text-sm font-medium text-slate-700">
            {dict.text}
          </span>
          <span className="rounded-full border border-white/70 bg-white/85 px-2.5 py-1 text-[0.68rem] font-semibold tracking-[0.14em] text-slate-500">
            {dict.date}
          </span>
          <ArrowUpRight className="h-4 w-4 text-slate-500" />
        </a>
      )}

      {/* Desktop navigation */}
      <div className="fixed top-4 right-4 z-50 hidden items-center gap-3 lg:flex">
        <a
          href={`/${currentLang}/manual/`}
          className={buttonClass}
          target="_blank"
          rel="noopener noreferrer"
        >
          <Book className="w-4 h-4" />
          <span className="uppercase font-mono">Docs</span>
        </a>
        <a
          href="https://github.com/lablup/mlxcel-releases"
          className={buttonClass}
          target="_blank"
          rel="noopener noreferrer"
        >
          <span className="uppercase font-mono">GitHub</span>
        </a>
        <button onClick={toggleLanguage} className={buttonClass}>
          <Globe className="w-4 h-4" />
          <span className="uppercase font-mono">
            {currentLang === "ko" ? "EN" : "KO"}
          </span>
        </button>
      </div>

      {/* Mobile navigation */}
      <div className="fixed top-4 right-4 z-50 flex items-center gap-2 lg:hidden">
        <button
          onClick={toggleLanguage}
          className={mobileButtonClass}
          aria-label={currentLang === "ko" ? "Switch to English" : "Switch to Korean"}
        >
          <Globe className="h-4 w-4" />
          <span className="font-mono uppercase">
            {currentLang === "ko" ? "EN" : "KO"}
          </span>
        </button>
        <button
          onClick={() => setMobileMenuOpen((open) => !open)}
          className={mobileButtonClass}
          aria-expanded={mobileMenuOpen}
          aria-controls="landing-mobile-menu"
        >
          {mobileMenuOpen ? (
            <X className="h-4 w-4" />
          ) : (
            <Menu className="h-4 w-4" />
          )}
          <span className="font-medium">
            {currentLang === "ko" ? "메뉴" : "Menu"}
          </span>
        </button>
      </div>

      {dict.visible && (
        <a
          href={dict.href}
          target="_blank"
          rel="noopener noreferrer"
          className={mobileAnnouncementClass}
          aria-label={`${dict.text} (${dict.date})`}
        >
          <div className="flex min-w-0 items-center gap-2">
            <span className="shrink-0 rounded-full bg-[linear-gradient(135deg,rgba(251,191,36,0.18),rgba(251,146,60,0.12))] px-2 py-1 text-[0.62rem] font-semibold uppercase tracking-[0.18em] text-amber-700">
              {dict.tag}
            </span>
            <span className="ko-keep-all truncate text-[0.82rem] font-medium text-slate-700 sm:text-sm">
              {dict.text}
            </span>
          </div>
          <div className="flex shrink-0 items-center gap-2">
            <span className="rounded-full border border-white/70 bg-white/85 px-2 py-1 text-[0.62rem] font-semibold tracking-[0.14em] text-slate-500">
              {dict.date}
            </span>
            <ArrowUpRight className="h-4 w-4 text-slate-500" />
          </div>
        </a>
      )}

      {mobileMenuOpen && (
        <div
          id="landing-mobile-menu"
          className="fixed inset-x-4 top-[4.5rem] z-40 rounded-2xl border border-slate-200 bg-white/96 p-3 shadow-[0_30px_80px_-28px_rgba(15,23,42,0.22)] backdrop-blur-xl lg:hidden"
        >
          <div className="grid gap-2">
            <a
              href={`/${currentLang}/manual/`}
              className="flex items-center gap-3 rounded-xl border border-transparent bg-slate-900/[0.03] px-4 py-3 text-sm text-slate-700 transition-colors hover:border-cyan-400/20 hover:bg-cyan-50"
              target="_blank"
              rel="noopener noreferrer"
              onClick={() => setMobileMenuOpen(false)}
            >
              <Book className="h-4 w-4 text-cyan-600" />
              <span>Docs</span>
            </a>
            <a
              href="https://github.com/lablup/mlxcel-releases"
              className="flex items-center gap-3 rounded-xl border border-transparent bg-slate-900/[0.03] px-4 py-3 text-sm text-slate-700 transition-colors hover:border-cyan-400/20 hover:bg-cyan-50"
              target="_blank"
              rel="noopener noreferrer"
              onClick={() => setMobileMenuOpen(false)}
            >
              <span>GitHub</span>
            </a>
          </div>
        </div>
      )}
    </>
  );
}
