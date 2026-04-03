import Link from "next/link";
import { NavTabs } from "./NavTabs";

export default function DashboardLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <div className="bg-gray-100 min-h-screen">
      <nav className="bg-gray-900 px-6 py-3">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-6">
            <Link href="/" className="text-lg font-bold text-white">
              Prompt2Policy
            </Link>
            <NavTabs />
          </div>
          <Link
            href="/iterations"
            className="text-gray-300 text-sm font-medium hover:text-white"
          >
            Iterations
          </Link>
        </div>
      </nav>
      <main className="max-w-6xl mx-auto px-6 py-8">{children}</main>
    </div>
  );
}
