"use client";

import { useState } from "react";

export default function Tooltip({ content }: { content: React.ReactNode }) {
  const [open, setOpen] = useState(false);

  return (
    <span
      className="relative inline-flex ml-1 align-middle"
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
    >
      <span
        className="inline-flex items-center justify-center w-4 h-4 rounded-full bg-gray-200 text-gray-500 text-[10px] font-bold cursor-help"
        aria-label="Help"
      >
        ?
      </span>
      {open && (
        <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1.5 z-50 w-64 bg-gray-900 text-gray-100 text-xs rounded-lg shadow-lg px-3 py-2 leading-relaxed pointer-events-none">
          {content}
          <div className="absolute top-full left-1/2 -translate-x-1/2 w-0 h-0 border-x-4 border-x-transparent border-t-4 border-t-gray-900" />
        </div>
      )}
    </span>
  );
}
