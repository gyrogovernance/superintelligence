import { useState, useEffect } from 'react';
import type { ProgramResponse } from '../types';
import * as api from '../api';
import { ConfirmModal } from './ConfirmModal';

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
  const [verifyStatus, setVerifyStatus] = useState<string | null>(null);
  const [signBundleOnSync, setSignBundleOnSync] = useState<boolean>(false);
  const [hasSigningKey, setHasSigningKey] = useState<boolean>(false);
  const [deleteConfirm, setDeleteConfirm] = useState(false);

  useEffect(() => {
    async function loadSignBundleSetting() {
      try {
        const config = await api.getSignBundleOnSync();
        setSignBundleOnSync(config.sign_bundle_on_sync);
        setHasSigningKey(config.has_signing_key);
      } catch (err) {
        // Ignore errors
      }
    }
    loadSignBundleSetting();
  }, []);

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

  const handleDelete = () => {
    if (!selectedSlug) return;
    setDeleteConfirm(true);
  };

  const handleDeleteConfirm = async () => {
    if (!selectedSlug) return;
    setDeleteConfirm(false);
    try {
      await api.deleteProgram(selectedSlug);
      window.location.reload();
    } catch (err) {
      alert(`Error: ${err instanceof Error ? err.message : 'Delete failed'}`);
    }
  };

  const handleVerifyBundle = async () => {
    if (!selectedSlug) return;
    try {
      setVerifyStatus('Verifying...');
      const result = await api.verifyBundle(selectedSlug);
      await onProgramReload();
      if (result.verified) {
        setVerifyStatus('Verified');
      } else {
        setVerifyStatus(`Verification failed: ${result.status}`);
      }
      setTimeout(() => setVerifyStatus(null), 5000);
    } catch (err) {
      setVerifyStatus(`Error: ${err instanceof Error ? err.message : 'Verification failed'}`);
      setTimeout(() => setVerifyStatus(null), 5000);
    }
  };

  const handleSignBundleOnSyncChange = async (checked: boolean) => {
    try {
      const res = await api.setSignBundleOnSync(checked);
      setSignBundleOnSync(res.sign_bundle_on_sync);
      setHasSigningKey(res.has_signing_key);
    } catch (err) {
      alert(`Error: ${err instanceof Error ? err.message : 'Failed to update setting'}`);
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
              <h3 className="font-semibold text-sm text-gray-500 dark:text-gray-400 uppercase tracking-wide mb-3">
                Governance and Verification
              </h3>
              <div className="space-y-3">
                {program?.report && (
                  <div className="flex items-center justify-between p-3 rounded-lg bg-gray-50 dark:bg-gray-800/50">
                    <div>
                      <div className="font-medium">Programme ID</div>
                      <div className="text-sm text-gray-500 dark:text-gray-400 font-mono">
                        {program.report.program_id}
                      </div>
                    </div>
                  </div>
                )}
                <div className="flex items-center justify-between p-3 rounded-lg bg-gray-50 dark:bg-gray-800/50">
                  <div>
                    <div className="font-medium">Last bundle signed by</div>
                    <div className="text-sm text-gray-500 dark:text-gray-400 font-mono">
                      {program?.governance?.signer_fingerprint || 'Not signed'}
                    </div>
                  </div>
                </div>
                {program?.governance && (
                  <div className="flex items-center justify-between p-3 rounded-lg bg-gray-50 dark:bg-gray-800/50">
                    <div>
                      <div className="font-medium">Last bundle verification result</div>
                      <div className="text-sm text-gray-500 dark:text-gray-400">
                        {program.governance.status}
                      </div>
                    </div>
                  </div>
                )}
                <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-3">
                  <div className="flex items-center justify-between mb-2">
                    <div>
                      <div className="font-medium">Sign bundle on sync</div>
                      <div className="text-sm text-gray-500 dark:text-gray-400">
                        Automatically sign bundles when syncing
                      </div>
                    </div>
                    <label className="flex items-center cursor-pointer">
                      <input
                        type="checkbox"
                        checked={signBundleOnSync}
                        onChange={(e) => handleSignBundleOnSyncChange(e.target.checked)}
                        className="rounded text-indigo-600"
                      />
                    </label>
                  </div>
                  {signBundleOnSync && !hasSigningKey && (
                    <div className="text-xs text-amber-600 dark:text-amber-400 bg-amber-50 dark:bg-amber-950/30 p-2 rounded">
                      ⚠️ No signing key found in .aci/.config.json. Bundles will not be signed.
                    </div>
                  )}
                </div>
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium">Verify latest bundle</div>
                    <div className="text-sm text-gray-500 dark:text-gray-400">
                      Download and verify the latest bundle
                    </div>
                  </div>
                  <button
                    type="button"
                    className="btn btn-primary btn-sm"
                    onClick={handleVerifyBundle}
                  >
                    Verify
                  </button>
                </div>
                {verifyStatus && (
                  <div className={`text-sm ${verifyStatus.startsWith('Verified') ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                    {verifyStatus}
                  </div>
                )}
              </div>
            </div>

            {program?.ecology && (
              <div>
                <h3 className="font-semibold text-sm text-gray-500 dark:text-gray-400 uppercase tracking-wide mb-3">
                  Ecology Capacity
                </h3>
                <div className="space-y-3">
                  <div className="p-3 rounded-lg bg-gray-50 dark:bg-gray-800/50 space-y-3">
                    <div className="grid grid-cols-3 gap-3">
                      <div>
                        <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">Total Capacity</div>
                        <div className="font-medium font-mono text-sm">
                          {program.ecology.total_capacity_MU.toLocaleString()} MU
                        </div>
                      </div>
                      <div>
                        <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">Used Capacity</div>
                        <div className="font-medium font-mono text-sm text-indigo-600 dark:text-indigo-400">
                          {program.ecology.used_capacity_MU.toLocaleString()} MU
                        </div>
                      </div>
                      <div>
                        <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">Free Capacity</div>
                        <div className="font-medium font-mono text-sm text-emerald-600 dark:text-emerald-400">
                          {program.ecology.free_capacity_MU.toLocaleString()} MU
                        </div>
                      </div>
                    </div>
                    {program.ecology.total_capacity_MU > 0 && (() => {
                      const usagePercent = (program.ecology.used_capacity_MU / program.ecology.total_capacity_MU) * 100;
                      const displayPercent = usagePercent < 0.01 
                        ? usagePercent.toExponential(2) + '%'
                        : usagePercent.toFixed(2) + '%';
                      return (
                        <div className="space-y-1">
                          <div className="flex items-center justify-between text-xs">
                            <span className="text-gray-500 dark:text-gray-400">Usage</span>
                            <span className="font-medium">
                              {displayPercent}
                            </span>
                          </div>
                          <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                            <div
                              className="bg-indigo-600 dark:bg-indigo-400 h-2 rounded-full transition-all"
                              style={{
                                width: `${Math.max(1, (program.ecology.used_capacity_MU / program.ecology.total_capacity_MU) * 100)}%`,
                              }}
                            />
                          </div>
                        </div>
                      );
                    })()}
                  </div>
                  {program.ecology.shells && program.ecology.shells.length > 0 && (
                    <div className="rounded-lg bg-gray-50 dark:bg-gray-800/50 overflow-hidden border border-gray-200 dark:border-gray-700">
                      <div className="p-3 border-b border-gray-200 dark:border-gray-700">
                        <div className="font-medium text-sm">Shells ({program.ecology.shells.length})</div>
                      </div>
                      <div className="overflow-x-auto max-h-60 overflow-y-auto">
                        <table className="w-full text-sm">
                          <thead className="bg-gray-100 dark:bg-gray-900/50 text-xs uppercase text-gray-500 dark:text-gray-400 sticky top-0 z-10">
                            <tr>
                              <th className="px-3 py-2 text-left bg-gray-100 dark:bg-gray-900">Header</th>
                              <th className="px-3 py-2 text-left bg-gray-100 dark:bg-gray-900">Seal</th>
                              <th className="px-3 py-2 text-right bg-gray-100 dark:bg-gray-900">Used</th>
                              <th className="px-3 py-2 text-right bg-gray-100 dark:bg-gray-900">Total</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                            {program.ecology.shells.map((shell, idx) => {
                              const sealShort = shell.seal.length >= 12 
                                ? `${shell.seal.slice(0, 6)}...${shell.seal.slice(-6)}`
                                : shell.seal;
                              return (
                                <tr key={idx} className="hover:bg-gray-100 dark:hover:bg-gray-800/50">
                                  <td className="px-3 py-2 font-mono text-xs whitespace-nowrap">{shell.header}</td>
                                  <td className="px-3 py-2 font-mono text-xs text-gray-600 dark:text-gray-400" title={shell.seal}>
                                    {sealShort}
                                  </td>
                                  <td className="px-3 py-2 text-right font-mono text-xs whitespace-nowrap">
                                    {shell.used_capacity_MU.toLocaleString()} MU
                                  </td>
                                  <td className="px-3 py-2 text-right font-mono text-xs text-gray-500 dark:text-gray-400 whitespace-nowrap">
                                    {shell.total_capacity_MU.toLocaleString()} MU
                                  </td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

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

      <ConfirmModal
        isOpen={deleteConfirm}
        title="Delete Program"
        message={`Are you sure you want to delete program "${selectedSlug}"? This will remove the local program file and all compiled artifacts. Previously published bundles are not affected. This action cannot be undone.`}
        confirmLabel="Delete"
        variant="danger"
        onConfirm={handleDeleteConfirm}
        onCancel={() => setDeleteConfirm(false)}
      />
    </div>
  );
}

