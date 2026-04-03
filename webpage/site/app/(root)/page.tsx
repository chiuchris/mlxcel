import { redirect } from "next/navigation";

export default function RootPage() {
  // Default redirect to English page
  // Browser language detection happens client-side on the /en page
  redirect("/en");
}
