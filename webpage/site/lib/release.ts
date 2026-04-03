export interface ReleaseAsset {
  name: string;
  browser_download_url: string;
  size: number;
}

export interface ReleaseData {
  tag_name: string;
  published_at: string;
  assets: ReleaseAsset[];
  html_url: string;
}

const RELEASE_URL =
  "https://api.github.com/repos/lablup/mlxcel-releases/releases/latest";

export async function fetchLatestRelease(): Promise<ReleaseData | null> {
  try {
    const res = await fetch(RELEASE_URL, {
      headers: {
        Accept: "application/vnd.github.v3+json",
        ...(process.env.GITHUB_TOKEN
          ? { Authorization: `Bearer ${process.env.GITHUB_TOKEN}` }
          : {}),
      },
    });

    if (!res.ok) {
      console.error(
        `Failed to fetch release: ${String(res.status)} ${res.statusText}`
      );
      return null;
    }

    return (await res.json()) as ReleaseData;
  } catch (err) {
    console.error("Failed to fetch latest release:", err);
    return null;
  }
}
