import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Fish Classifier",
  description: "Upload a fish image to get a species prediction",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body suppressHydrationWarning>{children}</body>
    </html>
  );
}
