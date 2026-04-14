"use client";

import { useEffect, useState } from "react";

export type OS = "macos" | "windows" | "linux" | "unknown";
export type Arch = "x86_64" | "arm64" | "unknown";

export interface Platform {
  os: OS;
  arch: Arch;
}

interface NavigatorUAData {
  getHighEntropyValues: (
    hints: string[]
  ) => Promise<{ architecture?: string; bitness?: string }>;
}

interface WebGLDebugInfoExtension {
  UNMASKED_RENDERER_WEBGL: number;
}

function detectOsFromUserAgent(userAgent: string): OS {
  if (userAgent.includes("mac")) return "macos";
  if (userAgent.includes("win")) return "windows";
  if (userAgent.includes("linux")) return "linux";
  return "unknown";
}

function detectArchFromUserAgent(userAgent: string): Arch {
  if (userAgent.includes("arm64") || userAgent.includes("aarch64")) {
    return "arm64";
  }
  if (
    userAgent.includes("x86_64") ||
    userAgent.includes("amd64") ||
    userAgent.includes("win64")
  ) {
    return "x86_64";
  }
  return "unknown";
}

async function refineArchWithClientHints(fallback: Arch): Promise<Arch> {
  if (!("userAgentData" in navigator)) return fallback;
  try {
    const uaData = (navigator as Navigator & { userAgentData: NavigatorUAData })
      .userAgentData;
    const values = await uaData.getHighEntropyValues([
      "architecture",
      "bitness",
    ]);
    if (values.architecture === "arm") return "arm64";
    if (values.architecture === "x86") return "x86_64";
  } catch {
    // Ignore — fall back to the user-agent guess.
  }
  return fallback;
}

function refineArchWithWebGL(fallback: Arch): Arch {
  try {
    const canvas = document.createElement("canvas");
    const gl =
      canvas.getContext("webgl") || canvas.getContext("experimental-webgl");
    if (!gl) return fallback;
    const typedGl = gl as WebGLRenderingContext;
    const debugInfo = typedGl.getExtension(
      "WEBGL_debug_renderer_info"
    ) as WebGLDebugInfoExtension | null;
    if (!debugInfo) return fallback;
    const renderer = typedGl.getParameter(
      debugInfo.UNMASKED_RENDERER_WEBGL
    ) as string;
    if (renderer.includes("Apple M") || renderer.includes("Apple GPU")) {
      return "arm64";
    }
    return "x86_64";
  } catch {
    return fallback;
  }
}

export function useOS(): OS {
  // Initialize synchronously from the user agent during the first render so
  // there is no effect-only setState. On the server, `navigator` is absent
  // and we default to "unknown"; the client render then produces the real
  // value on mount without an extra effect-triggered re-render.
  const [os] = useState<OS>(() => {
    if (typeof navigator === "undefined") return "unknown";
    return detectOsFromUserAgent(navigator.userAgent.toLowerCase());
  });
  return os;
}

export function usePlatform(): Platform {
  const [platform, setPlatform] = useState<Platform>(() => {
    if (typeof navigator === "undefined") {
      return { os: "unknown", arch: "unknown" };
    }
    const ua = navigator.userAgent.toLowerCase();
    return {
      os: detectOsFromUserAgent(ua),
      arch: detectArchFromUserAgent(ua),
    };
  });

  useEffect(() => {
    // Only the arch refinement below needs async work (client hints /
    // WebGL probe), so treat it as an external-system sync.
    let cancelled = false;

    (async () => {
      const ua = navigator.userAgent.toLowerCase();
      let detectedArch = detectArchFromUserAgent(ua);
      const os = detectOsFromUserAgent(ua);

      detectedArch = await refineArchWithClientHints(detectedArch);

      if (os === "macos" && detectedArch === "unknown") {
        detectedArch = refineArchWithWebGL(detectedArch);
      }

      if (!cancelled) {
        setPlatform({ os, arch: detectedArch });
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  return platform;
}
