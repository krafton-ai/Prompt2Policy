"use client";

import { useEffect, useRef } from "react";
import katex from "katex";
import "katex/dist/katex.min.css";
import type { RewardTerm } from "@/lib/api";

interface Props {
  latex: string;
  terms: RewardTerm[] | Record<string, string>;
}

/** Try to render a string as KaTeX; return the HTML or null on failure. */
function tryRenderLatex(src: string): string | null {
  try {
    const stripped = src.replace(/^\$+|\$+$/g, "").trim();
    if (!stripped) return null;
    return katex.renderToString(stripped, {
      throwOnError: false,
      displayMode: false,
    });
  } catch {
    return null;
  }
}

/** Normalize terms to structured array format (backward compat). */
function normalizeTerms(
  terms: RewardTerm[] | Record<string, string>,
): RewardTerm[] {
  if (Array.isArray(terms)) return terms;
  // Legacy dict format: convert to structured array
  return Object.entries(terms).map(([name, value]) => {
    const idx = value.indexOf("\n");
    if (idx === -1) return { name, description: value };
    return {
      name,
      description: value.slice(0, idx),
      latex: value.slice(idx + 1).trim(),
    };
  });
}

export default function RewardFormula({ latex, terms }: Props) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (ref.current && latex) {
      const stripped = latex.replace(/^\$+|\$+$/g, "").trim();
      katex.render(stripped, ref.current, {
        throwOnError: false,
        displayMode: true,
      });
    }
  }, [latex]);

  const normalizedTerms = normalizeTerms(terms);

  return (
    <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-5">
      <h2 className="text-sm font-semibold text-gray-900 mb-3">
        Reward Function
      </h2>
      {latex && (
        <div ref={ref} className="mb-4 overflow-x-auto py-2" />
      )}
      {normalizedTerms.length > 0 && (
        <div className="space-y-2">
          {normalizedTerms.map((term) => {
            const eqHtml = term.latex ? tryRenderLatex(term.latex) : null;
            return (
              <div key={term.name}>
                <div className="flex items-start gap-2 text-sm">
                  <code className="bg-gray-100 px-1.5 py-0.5 rounded text-xs font-mono text-gray-800 shrink-0">
                    {term.name}
                  </code>
                  <span className="text-gray-600">
                    {term.description ?? ""}
                  </span>
                </div>
                {eqHtml ? (
                  <div
                    className="ml-6 mt-0.5 text-xs text-gray-500 overflow-x-auto"
                    dangerouslySetInnerHTML={{ __html: eqHtml }}
                  />
                ) : term.latex ? (
                  <code className="ml-6 mt-0.5 block text-xs text-gray-500 font-mono">
                    {term.latex}
                  </code>
                ) : null}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
