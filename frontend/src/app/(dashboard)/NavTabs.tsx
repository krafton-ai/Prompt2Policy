"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const tabs = [
  { href: "/e2e", label: "E2E" },
  { href: "/benchmark", label: "Benchmark" },
  { href: "/trash", label: "Trash" },
  { href: "/monitor", label: "Monitor" },
] as const;

export function NavTabs() {
  const pathname = usePathname();

  return (
    <div className="flex gap-1">
      {tabs.map((tab) => {
        const active = pathname.startsWith(tab.href);
        return (
          <Link
            key={tab.href}
            href={tab.href}
            className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
              active
                ? "bg-gray-700 text-white"
                : "text-gray-400 hover:text-white hover:bg-gray-800"
            }`}
          >
            {tab.label}
          </Link>
        );
      })}
    </div>
  );
}
