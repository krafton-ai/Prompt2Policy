"use client";

import { useState, useEffect, useCallback } from "react";
import type { NodeResponse, NodeCheckResponse } from "@/lib/scheduler-api";
import { checkNode, removeNode, updateNode, setupNode } from "@/lib/scheduler-api";

const AUTO_CHECK_INTERVAL = 60_000; // 60 seconds

export default function NodeCard({
  node,
  onRemoved,
  onUpdated,
}: {
  node: NodeResponse;
  onRemoved: () => void;
  onUpdated: () => void;
}) {
  const [checking, setChecking] = useState(false);
  const [checkResult, setCheckResult] = useState<NodeCheckResponse | null>(
    null,
  );
  const [removing, setRemoving] = useState(false);
  const [editing, setEditing] = useState(false);
  const [maxCores, setMaxCores] = useState(node.max_cores);
  const [saving, setSaving] = useState(false);
  const [settingUp, setSettingUp] = useState(false);
  const [setupResult, setSetupResult] = useState<string | null>(null);
  const [toggling, setToggling] = useState(false);

  const handleCheck = useCallback(async () => {
    setChecking(true);
    try {
      const result = await checkNode(node.node_id);
      setCheckResult(result);
    } finally {
      setChecking(false);
    }
  }, [node.node_id]);

  // Auto-check on mount and every 60 seconds
  useEffect(() => {
    handleCheck();
    const timer = setInterval(handleCheck, AUTO_CHECK_INTERVAL);
    return () => clearInterval(timer);
  }, [handleCheck]);

  const handleRemove = async () => {
    if (!confirm(`Remove node "${node.node_id}"?`)) return;
    setRemoving(true);
    try {
      await removeNode(node.node_id);
      onRemoved();
    } finally {
      setRemoving(false);
    }
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      await updateNode(node.node_id, { max_cores: maxCores });
      setEditing(false);
      onUpdated();
    } finally {
      setSaving(false);
    }
  };

  const handleToggleEnabled = async () => {
    setToggling(true);
    try {
      await updateNode(node.node_id, { enabled: !node.enabled });
      onUpdated();
    } finally {
      setToggling(false);
    }
  };

  const handleSetup = async () => {
    setSettingUp(true);
    setSetupResult(null);
    try {
      const result = await setupNode(node.node_id);
      if (result.ok) {
        setSetupResult("Setup complete");
        handleCheck(); // re-check to update uv status
      } else {
        setSetupResult(result.error ?? "Setup failed");
      }
    } catch {
      setSetupResult("Setup request failed");
    } finally {
      setSettingUp(false);
    }
  };

  return (
    <div className={`rounded-lg border p-4 shadow-sm ${node.enabled ? "border-gray-200 bg-white" : "border-gray-200 bg-gray-50 opacity-60"}`}>
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <button
            onClick={handleToggleEnabled}
            disabled={toggling}
            className={`relative inline-flex h-5 w-9 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none ${
              node.enabled ? "bg-blue-600" : "bg-gray-300"
            } disabled:opacity-50`}
            title={node.enabled ? "Disable node" : "Enable node"}
          >
            <span
              className={`pointer-events-none inline-block h-4 w-4 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
                node.enabled ? "translate-x-4" : "translate-x-0"
              }`}
            />
          </button>
          <h3 className="font-semibold text-gray-900">{node.node_id}</h3>
        </div>
        <span
          className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${
            checkResult == null
              ? "bg-gray-100 text-gray-500"
              : checkResult.online
                ? "bg-green-100 text-green-800"
                : "bg-red-100 text-red-800"
          }`}
        >
          {checkResult == null ? "checking..." : checkResult.online ? "online" : "offline"}
        </span>
      </div>

      <div className="text-sm text-gray-600 space-y-1">
        <p>
          {node.user}@{node.host}:{node.port}
        </p>
        <p className="font-mono text-xs">{node.base_dir}</p>
        {editing ? (
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-500">Max cores:</span>
            <input
              type="number"
              min={1}
              value={maxCores}
              onChange={(e) => setMaxCores(Math.max(1, Number(e.target.value)))}
              className="w-16 rounded border border-gray-300 px-2 py-0.5 text-sm font-mono"
            />
            <button
              onClick={handleSave}
              disabled={saving}
              className="px-2 py-0.5 text-xs rounded bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50"
            >
              {saving ? "..." : "Save"}
            </button>
            <button
              onClick={() => { setEditing(false); setMaxCores(node.max_cores); }}
              className="px-2 py-0.5 text-xs rounded text-gray-500 hover:text-gray-700"
            >
              Cancel
            </button>
          </div>
        ) : (
          <p
            className="cursor-pointer hover:text-blue-600"
            onClick={() => setEditing(true)}
            title="Click to edit"
          >
            Cores: {node.used_cores} / {node.max_cores}
          </p>
        )}
        {checkResult?.gpu && (
          <p className="text-xs text-purple-700">
            GPU: {checkResult.gpu}
            <span className={`ml-2 inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium ${
              checkResult.mps_active ? "bg-green-100 text-green-700" : "bg-gray-100 text-gray-500"
            }`}>MPS {checkResult.mps_active ? "ON" : "OFF"}</span>
          </p>
        )}
        {checkResult?.online && !checkResult.gpu && (
          <p className="text-xs text-gray-400">GPU: none</p>
        )}
        {checkResult?.uv_available === false && checkResult.online && (
          <p className="text-amber-600 text-xs">uv not found on remote</p>
        )}
        {setupResult && (
          <p className={`text-xs ${setupResult === "Setup complete" ? "text-green-600" : "text-red-600"}`}>
            {setupResult}
          </p>
        )}
        {checkResult?.error && (
          <p className="text-red-600 text-xs">{checkResult.error}</p>
        )}
      </div>

      <div className="flex gap-2 mt-3">
        <button
          onClick={handleCheck}
          disabled={checking}
          className="px-3 py-1 text-xs rounded-md bg-gray-100 hover:bg-gray-200 text-gray-700 disabled:opacity-50"
        >
          {checking ? "Checking..." : "Check"}
        </button>
        {checkResult?.online && checkResult.uv_available === false && (
          <button
            onClick={handleSetup}
            disabled={settingUp}
            className="px-3 py-1 text-xs rounded-md bg-blue-50 hover:bg-blue-100 text-blue-700 disabled:opacity-50"
          >
            {settingUp ? "Setting up..." : "Setup"}
          </button>
        )}
        <button
          onClick={handleRemove}
          disabled={removing}
          className="px-3 py-1 text-xs rounded-md bg-red-50 hover:bg-red-100 text-red-700 disabled:opacity-50"
        >
          Remove
        </button>
      </div>
    </div>
  );
}
