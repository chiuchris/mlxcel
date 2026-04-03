import type { NextConfig } from "next";
import path from "path";

const nextConfig: NextConfig = {
  output: "export",
  trailingSlash: true,
  images: {
    unoptimized: true,
  },
  // Fix workspace root detection when nested in a monorepo
  turbopack: {
    root: path.resolve(__dirname),
  },
};

export default nextConfig;
