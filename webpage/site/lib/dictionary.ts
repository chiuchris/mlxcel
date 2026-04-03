import { en } from "@/dictionaries/en";
import { ko } from "@/dictionaries/ko";

export const dictionaries = {
  en,
  ko,
};

export type Locale = keyof typeof dictionaries;

export const getDictionary = (locale: Locale) =>
  dictionaries[locale] ?? dictionaries.en;
