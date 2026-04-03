"use client";

import { useEffect, useState } from "react";

export type OS = "macos" | "windows" | "linux" | "unknown";
export type Arch = "x86_64" | "arm64" | "unknown";

export interface Platform {
  os: OS;
  arch: Arch;
}

export function useOS(): OS {
  const [os, setOs] = useState<OS>("unknown");

  useEffect(() => {
    const userAgent = window.navigator.userAgent.toLowerCase();
    if (userAgent.includes("mac")) {
      setOs("macos");
    } else if (userAgent.includes("win")) {
      setOs("windows");
    } else if (userAgent.includes("linux")) {
      setOs("linux");
    } else {
      setOs("unknown");
    }
  }, []);

  return os;
}

export function usePlatform() {
  const [platform, setPlatform] = useState<Platform>({
    os: "unknown",
    arch: "unknown",
  });

  useEffect(() => {
    const userAgent = window.navigator.userAgent.toLowerCase();
    let os: OS = "unknown";
    let arch: Arch = "unknown";

    // Detect OS
    if (userAgent.includes("mac")) os = "macos";
    else if (userAgent.includes("win")) os = "windows";
    else if (userAgent.includes("linux")) os = "linux";

    // Basic Arch Detection from UserAgent
    if (userAgent.includes("arm64") || userAgent.includes("aarch64")) {
      arch = "arm64";
    } else if (
      userAgent.includes("x86_64") ||
      userAgent.includes("amd64") ||
      userAgent.includes("win64")
    ) {
      arch = "x86_64";
    }

    // Special handling for Apple Silicon & Modern Browsers
    const detectHighEntropyArch = async () => {
      let detectedArch = arch;

      // 1. Try Client Hints API (Modern Chrome/Edge)
      if ("userAgentData" in navigator) {
        try {
          const uaData = (navigator as any).userAgentData;
          const values = await uaData.getHighEntropyValues([
            "architecture",
            "bitness",
          ]);
          if (values.architecture === "arm") {
            detectedArch = "arm64";
          } else if (values.architecture === "x86") {
            detectedArch = "x86_64";
          }
        } catch (e) {
          // Ignore error
        }
      }

      // 2. WebGL Vendor Trick (For Safari/Mac)
      if (os === "macos" && detectedArch === "unknown") {
        try {
          const canvas = document.createElement("canvas");
          const gl =
            canvas.getContext("webgl") ||
            canvas.getContext("experimental-webgl");
          if (gl) {
            const debugInfo = (gl as any).getExtension(
              "WEBGL_debug_renderer_info"
            );
            if (debugInfo) {
              const renderer = (gl as any).getParameter(
                debugInfo.UNMASKED_RENDERER_WEBGL
              );
              if (
                renderer.includes("Apple M") ||
                renderer.includes("Apple GPU")
              ) {
                detectedArch = "arm64";
              } else {
                detectedArch = "x86_64";
              }
            }
          }
        } catch (e) {
          // Ignore
        }
      }

      setPlatform({ os, arch: detectedArch });
    };

    detectHighEntropyArch();
  }, []);

  return platform;
}
