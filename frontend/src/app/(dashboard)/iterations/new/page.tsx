"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { startSession } from "@/lib/api";

export default function NewIterationPage() {
  const router = useRouter();
  const [prompt, setPrompt] = useState("");
  const [timesteps, setTimesteps] = useState(10_000_000);
  const [seed, setSeed] = useState(1);
  const [maxIterations, setMaxIterations] = useState(5);
  const [passThreshold, setPassThreshold] = useState(0.7);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setSubmitting(true);
    setError("");
    try {
      const { session_id } = await startSession({
        prompt,
        total_timesteps: timesteps,
        seed,
        max_iterations: maxIterations,
        pass_threshold: passThreshold,
        num_envs: 1,
      });
      router.push(`/e2e/${session_id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start session");
      setSubmitting(false);
    }
  }

  return (
    <div className="max-w-lg mx-auto">
      <h1 className="text-2xl font-bold text-gray-900 mb-6">
        New Experiment
      </h1>

      <form onSubmit={handleSubmit} className="space-y-5">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Prompt
          </label>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="e.g., Run forward as fast as possible"
            rows={3}
            className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
          <p className="text-xs text-gray-400 mt-1">
            Describes the desired locomotion behavior for reward generation.
          </p>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Total Timesteps
            </label>
            <input
              type="number"
              value={timesteps}
              onChange={(e) => setTimesteps(Number(e.target.value))}
              min={10000}
              step={10000}
              className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Seed
            </label>
            <input
              type="number"
              value={seed}
              onChange={(e) => setSeed(Number(e.target.value))}
              min={0}
              className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Max Iterations
            </label>
            <input
              type="number"
              value={maxIterations}
              onChange={(e) => setMaxIterations(Number(e.target.value))}
              min={1}
              className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Pass Threshold
            </label>
            <input
              type="number"
              value={passThreshold}
              onChange={(e) => setPassThreshold(Number(e.target.value))}
              min={0}
              max={1}
              step={0.1}
              className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
        </div>

        {error && <p className="text-sm text-red-600">{error}</p>}

        <button
          type="submit"
          disabled={submitting}
          className="w-full bg-blue-600 text-white py-2.5 rounded-lg font-medium text-sm hover:bg-blue-700 disabled:bg-blue-300 transition-colors"
        >
          {submitting ? "Starting..." : "Start Loop"}
        </button>
      </form>
    </div>
  );
}
