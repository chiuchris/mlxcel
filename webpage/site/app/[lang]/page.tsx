import { Hero } from "@/components/hero";
import { Highlights } from "@/components/highlights";
import { FeatureStory } from "@/components/feature-story";
import { Showcase } from "@/components/showcase";
import { Downloads } from "@/components/downloads";
import { BrewInstall } from "@/components/brew-install";
import { EnterpriseCta } from "@/components/enterprise-cta";
import { Footer } from "@/components/footer";
import { getDictionary, Locale } from "@/lib/dictionary";
import { LanguageSwitcher } from "@/components/language-switcher";
import { fetchLatestRelease } from "@/lib/release";

export function generateStaticParams() {
  return [{ lang: "en" }, { lang: "ko" }];
}

interface PageProps {
  params: Promise<{ lang: Locale }>;
}

export default async function Home({ params }: PageProps) {
  const { lang } = await params;
  const dict = getDictionary(lang);
  const release = await fetchLatestRelease();

  return (
    <main className="min-h-screen bg-[#f7fbfe] text-slate-900 selection:bg-cyan-200/70">
      <LanguageSwitcher dict={dict.announcement} />
      <Hero dict={dict.hero} release={release} />
      <Highlights dict={dict.highlights} />
      <FeatureStory dict={dict.story} />
      <Showcase dict={dict.showcase} />
      <Downloads dict={dict.downloads} release={release} />
      <BrewInstall dict={dict.brew} />
      <EnterpriseCta dict={dict.enterprise} />
      <Footer dict={dict.footer} lang={lang} />
    </main>
  );
}
