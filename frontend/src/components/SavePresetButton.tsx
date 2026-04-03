"use client";

import { useState } from "react";
import { Save, X } from "lucide-react";

interface SavePresetButtonProps {
  onSave: (name: string) => Promise<void>;
}

export default function SavePresetButton({ onSave }: SavePresetButtonProps) {
  const [show, setShow] = useState(false);
  const [name, setName] = useState("");
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState(false);

  async function handleSubmit() {
    const trimmed = name.trim();
    if (!trimmed) return;
    setError(false);
    try {
      await onSave(trimmed);
      setSaved(true);
      setTimeout(() => {
        setShow(false);
        setName("");
        setSaved(false);
      }, 1500);
    } catch {
      setError(true);
    }
  }

  if (!show) {
    return (
      <button
        type="button"
        onClick={() => setShow(true)}
        className="inline-flex items-center gap-1 px-3 py-1 rounded-full text-xs font-medium border border-gray-200 text-gray-500 hover:border-gray-300 hover:text-gray-700 transition-colors"
      >
        <Save size={12} />
        Save as Preset
      </button>
    );
  }

  return (
    <form
      onSubmit={(e) => { e.preventDefault(); handleSubmit(); }}
      className="inline-flex items-center gap-1"
    >
      <input
        autoFocus
        value={name}
        onChange={(e) => setName(e.target.value)}
        placeholder="Preset name"
        className={`w-28 px-2 py-1 rounded-lg border text-xs focus:outline-none focus:ring-1 ${error ? "border-red-400 focus:ring-red-400" : "border-gray-300 focus:ring-blue-400"}`}
        onKeyDown={(e) => { if (e.key === "Escape") { setShow(false); setName(""); } }}
      />
      <button
        type="submit"
        disabled={!name.trim() || saved}
        className="px-2 py-1 rounded-full text-xs font-medium bg-blue-600 text-white hover:bg-blue-700 disabled:bg-blue-300 transition-colors"
      >
        {saved ? "Saved!" : "Save"}
      </button>
      <button
        type="button"
        onClick={() => { setShow(false); setName(""); }}
        className="px-1 py-1 text-xs text-gray-400 hover:text-gray-600"
      >
        <X size={14} />
      </button>
    </form>
  );
}
