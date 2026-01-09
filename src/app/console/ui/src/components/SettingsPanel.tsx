import { useState } from 'react';
import type { ProgramResponse } from '../types';
import * as api from '../api';

interface SettingsPanelProps {
  program: ProgramResponse | null;
  selectedSlug: string | null;
  onProgramReload: () => void;
  unit: 'daily' | 'sprint';
  onUnitChange: (unit: 'daily' | 'sprint') => void;
}

export function SettingsPanel({ 
  program, 
  selectedSlug, 
  onProgramReload,
  unit,
  onUnitChange,
}: SettingsPanelProps) {
  const [exportStatus, setExportStatus] = useState<string | null>(null);
  const [importStatus, setImportStatus] = useState<string | null>(null);
  const [syncStatus, setSyncStatus] = useState<string | null>(null);

  const handleExport = async () => {
    if (!selectedSlug) return;
    try {
      setExportStatus('Exporting...');
      const bundleUrl = api.bundleUrl(selectedSlug);
      window.open(bundleUrl, '_blank');
      setExportStatus('Export started');
      setTimeout(() => setExportStatus(null), 3000);
    } catch (err) {
      setExportStatus(`Error: ${err instanceof Error ? err.message : 'Export failed'}`);
    }
  };

  const handleImport = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    try {
      setImportStatus('Importing...');
      const text = await file.text();
      const data = JSON.parse(text);
      
      if (data.type === 'program-export' && data.slug) {
        setImportStatus(`Import for ${data.slug} - manual import not yet implemented`);
        setTimeout(() => setImportStatus(null), 5000);
      } else {
        setImportStatus('Invalid import file format');
        setTimeout(() => setImportStatus(null), 3000);
      }
    } catch (err) {
      setImportStatus(`Error: ${err instanceof Error ? err.message : 'Import failed'}`);
      setTimeout(() => setImportStatus(null), 3000);
    }
  };

  const handleSync = async () => {
    if (!selectedSlug) return;
    try {
      setSyncStatus('Syncing...');
      await api.syncProgram(selectedSlug);
      await onProgramReload();
      setSyncStatus('Sync complete');
      setTimeout(() => setSyncStatus(null), 3000);
    } catch (err) {
      setSyncStatus(`Error: ${err instanceof Error ? err.message : 'Sync failed'}`);
      setTimeout(() => setSyncStatus(null), 3000);
    }
  };

  const handleDelete = async () => {
    if (!selectedSlug) return;
    if (!confirm(`Are you sure you want to delete program "${selectedSlug}"? This cannot be undone.`)) {
      return;
    }
    try {
      await api.deleteProgram(selectedSlug);
      window.location.reload();
    } catch (err) {
      alert(`Error: ${err instanceof Error ? err.message : 'Delete failed'}`);
    }
  };

  return (
    <div className="card p-6 space-y-6">
      <h2 className="text-lg font-bold">Settings</h2>

      {!selectedSlug ? (
        <p className="text-gray-500 dark:text-gray-400">
          Select a program to access settings.
        </p>
      ) : (
        <>
          <div className="space-y-4">
            {/* Program Track (Unit) */}
            <div>
              <h3 className="font-semibold text-sm text-gray-500 dark:text-gray-400 uppercase tracking-wide mb-3">
                Program Track
              </h3>
              <div className="flex flex-wrap gap-3">
                <button
                  type="button"
                  onClick={() => onUnitChange('daily')}
                  className={`px-4 py-3 rounded-lg border-2 transition-colors ${
                    unit === 'daily'
                      ? 'border-blue-500 bg-blue-50 dark:bg-blue-950/50 text-blue-700 dark:text-blue-300'
                      : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
                  }`}
                >
                  <div className="font-semibold">Daily Prize Track</div>
                  <div className="text-sm text-gray-500 dark:text-gray-400">£120 per unit</div>
                </button>
                <button
                  type="button"
                  onClick={() => onUnitChange('sprint')}
                  className={`px-4 py-3 rounded-lg border-2 transition-colors ${
                    unit === 'sprint'
                      ? 'border-purple-500 bg-purple-50 dark:bg-purple-950/50 text-purple-700 dark:text-purple-300'
                      : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
                  }`}
                >
                  <div className="font-semibold">Sprint Stipend Track</div>
                  <div className="text-sm text-gray-500 dark:text-gray-400">£480 per unit</div>
                </button>
              </div>
            </div>

            <div>
              <h3 className="font-semibold text-sm text-gray-500 dark:text-gray-400 uppercase tracking-wide mb-3">
                Program Management
              </h3>
              <div className="space-y-3">
                <div className="flex items-center justify-between p-3 rounded-lg bg-gray-50 dark:bg-gray-800/50">
                  <div>
                    <div className="font-medium">Program Slug</div>
                    <div className="text-sm text-gray-500 dark:text-gray-400">{selectedSlug}</div>
                  </div>
                </div>
                {program?.report && (
                  <div className="flex items-center justify-between p-3 rounded-lg bg-gray-50 dark:bg-gray-800/50">
                    <div>
                      <div className="font-medium">Program ID</div>
                      <div className="text-sm text-gray-500 dark:text-gray-400 font-mono">
                        {program.report.program_id}
                      </div>
                    </div>
                  </div>
                )}
                {program?.last_synced && (
                  <div className="flex items-center justify-between p-3 rounded-lg bg-gray-50 dark:bg-gray-800/50">
                    <div>
                      <div className="font-medium">Last Synced</div>
                      <div className="text-sm text-gray-500 dark:text-gray-400">
                        {new Date(program.last_synced).toLocaleString()}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>

            <div>
              <h3 className="font-semibold text-sm text-gray-500 dark:text-gray-400 uppercase tracking-wide mb-3">
                File Operations
              </h3>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium">Export Bundle</div>
                    <div className="text-sm text-gray-500 dark:text-gray-400">
                      Download complete program bundle (ZIP)
                    </div>
                  </div>
                  <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    onClick={handleExport}
                  >
                    Export
                  </button>
                </div>
                {exportStatus && (
                  <div className="text-sm text-gray-500 dark:text-gray-400">{exportStatus}</div>
                )}

                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium">Import Bundle</div>
                    <div className="text-sm text-gray-500 dark:text-gray-400">
                      Import program from bundle file
                    </div>
                  </div>
                  <label className="btn btn-secondary btn-sm cursor-pointer">
                    Import
                    <input
                      type="file"
                      accept=".zip,.json"
                      className="hidden"
                      onChange={handleImport}
                    />
                  </label>
                </div>
                {importStatus && (
                  <div className="text-sm text-gray-500 dark:text-gray-400">{importStatus}</div>
                )}
              </div>
            </div>

            <div>
              <h3 className="font-semibold text-sm text-gray-500 dark:text-gray-400 uppercase tracking-wide mb-3">
                Sync & Verification
              </h3>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium">Sync Program</div>
                    <div className="text-sm text-gray-500 dark:text-gray-400">
                      Recompile program and update report
                    </div>
                  </div>
                  <button
                    type="button"
                    className="btn btn-primary btn-sm"
                    onClick={handleSync}
                  >
                    Sync Now
                  </button>
                </div>
                {syncStatus && (
                  <div className="text-sm text-gray-500 dark:text-gray-400">{syncStatus}</div>
                )}

                {program?.report?.compilation && (
                  <div className="p-3 rounded-lg bg-gray-50 dark:bg-gray-800/50 space-y-2">
                    <div className="font-medium text-sm">Verification Hashes</div>
                    <div className="text-xs font-mono break-all text-gray-600 dark:text-gray-400">
                      <div>Bytes: {program.report.compilation.hashes.bytes_sha256}</div>
                      <div>Events: {program.report.compilation.hashes.events_sha256}</div>
                    </div>
                  </div>
                )}
              </div>
            </div>

            <div>
              <h3 className="font-semibold text-sm text-red-600 dark:text-red-400 uppercase tracking-wide mb-3">
                Danger Zone
              </h3>
              <div className="flex items-center justify-between p-3 rounded-lg bg-red-50 dark:bg-red-950/30 border border-red-200 dark:border-red-800">
                <div>
                  <div className="font-medium text-red-800 dark:text-red-200">Delete Program</div>
                  <div className="text-sm text-red-600 dark:text-red-400">
                    Permanently delete this program and all its data
                  </div>
                </div>
                <button
                  type="button"
                  className="btn btn-danger btn-sm"
                  onClick={handleDelete}
                >
                  Delete
                </button>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

