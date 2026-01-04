import { Stepper } from './Stepper';
import type { Report, EditableState, DomainKey, PrincipleKey } from '../types';

interface ReportPanelProps {
  report: Report | null;
  onDownloadBundle: () => void;
  notes: string;
  onNotesChange: (value: string) => void;
  editable: EditableState | null;
  onDomainChange: (domain: DomainKey, value: number) => void;
  onPrincipleChange: (principle: PrincipleKey, value: number) => void;
  onDomainInfo: (domain: string) => void;
  onPrincipleInfo: (principle: string) => void;
}

const STAGES = [
  {
    key: 'GOV',
    title: 'Governance',
    alignment: { code: 'GMT', name: 'Governance Management Traceability' },
    displacement: { code: 'GTD', name: 'Governance Traceability Displacement' },
  },
  {
    key: 'INFO',
    title: 'Information',
    alignment: { code: 'ICV', name: 'Information Curation Variety' },
    displacement: { code: 'IVD', name: 'Information Variety Displacement' },
  },
  {
    key: 'INFER',
    title: 'Inference',
    alignment: { code: 'IIA', name: 'Inference Interaction Accountability' },
    displacement: { code: 'IAD', name: 'Inference Accountability Displacement' },
  },
  {
    key: 'INTEL',
    title: 'Intelligence',
    alignment: { code: 'ICI', name: 'Intelligence Cooperation Integrity' },
    displacement: { code: 'IID', name: 'Intelligence Integrity Displacement' },
  },
] as const;

const DOMAINS: Array<{
  key: DomainKey;
  label: string;
  sublabel: string;
}> = [
  { key: 'economy', label: 'Economy', sublabel: 'CGM operations' },
  { key: 'employment', label: 'Employment', sublabel: 'Gyroscope work' },
  { key: 'education', label: 'Education', sublabel: 'THM capacities' },
];

export function ReportPanel({
  report,
  onDownloadBundle,
  notes,
  onNotesChange,
  editable,
  onDomainChange,
  onPrincipleChange,
  onDomainInfo,
  onPrincipleInfo,
}: ReportPanelProps) {
  if (!report || !editable) {
    return (
      <div className="card p-6">
        <h2 className="text-lg font-bold mb-2">Report</h2>
        <p className="text-gray-500 dark:text-gray-400">
          No report data. Create or sync a project to generate a report.
        </p>
      </div>
    );
  }

  const { accounting, compilation } = report;
  const { thm, gyroscope } = accounting;
  const domainTotal = editable.domain_counts.economy + editable.domain_counts.employment + editable.domain_counts.education;

  return (
    <div className="card p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-bold">Report</h2>
        <button type="button" className="btn btn-primary btn-sm" onClick={onDownloadBundle}>
          Download Bundle
        </button>
      </div>

      {/* Notes */}
      <div className="space-y-2">
        <h3 className="font-semibold text-sm text-gray-500 dark:text-gray-400 uppercase tracking-wide">
          Notes
        </h3>
        <textarea
          className="input w-full min-h-[80px] resize-y"
          placeholder="Add context or key observations for this project..."
          value={notes}
          onChange={(e) => onNotesChange(e.target.value)}
        />
        <p className="text-xs text-gray-400 dark:text-gray-500">
          Notes are for human context and do not affect the compiled ledger or apertures.
        </p>
      </div>

      {/* Domains */}
      <div className="space-y-4">
        <h3 className="font-semibold text-sm text-gray-500 dark:text-gray-400 uppercase tracking-wide">
          Domains
        </h3>
        <div className="space-y-4">
          {DOMAINS.map(({ key, label, sublabel }) => {
            const percentage = domainTotal > 0 ? ((editable.domain_counts[key] / domainTotal) * 100).toFixed(0) : '0';
            return (
              <div key={key} className="space-y-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="font-medium">{label}</span>
                    <button
                      type="button"
                      className="p-1 rounded hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
                      onClick={() => onDomainInfo(key)}
                      aria-label={`Info about ${label}`}
                    >
                      <svg className="w-4 h-4 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                    </button>
                  </div>
                  <Stepper
                    value={editable.domain_counts[key]}
                    label={label}
                    onChange={(value) => onDomainChange(key, value)}
                  />
                </div>
                <div className="text-xs text-gray-500 dark:text-gray-400">{sublabel}</div>
              </div>
            );
          })}
        </div>
        <div className="pt-2 border-t border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-500">Total Domain Count</span>
            <span className="font-bold">{domainTotal}</span>
          </div>
        </div>
      </div>

      {/* GGG Stages */}
      <div className="space-y-4">
        <h3 className="font-semibold text-sm text-gray-500 dark:text-gray-400 uppercase tracking-wide">
          Principles by Stage
        </h3>
        {STAGES.map(({ key, title, alignment, displacement }) => (
          <div key={key} className="p-4 rounded-lg bg-gray-50 dark:bg-gray-800/50 space-y-3">
            <h4 className="font-semibold text-sm text-gray-700 dark:text-gray-300">{title}</h4>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {/* Alignment */}
              <div className="p-3 rounded-lg bg-emerald-50 dark:bg-emerald-950/30">
                <div className="flex flex-col gap-3 sm:grid sm:grid-cols-[1fr_auto] sm:items-center">
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <span className="font-bold text-emerald-700 dark:text-emerald-300">
                        {alignment.code}
                      </span>
                      <button
                        type="button"
                        className="p-1 rounded hover:bg-emerald-200/50 dark:hover:bg-emerald-800/50 transition-colors"
                        onClick={() => onPrincipleInfo(alignment.code)}
                        aria-label={`Info about ${alignment.code}`}
                      >
                        <svg className="w-4 h-4 text-emerald-600 dark:text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                      </button>
                    </div>
                    <div className="text-xs text-emerald-700 dark:text-emerald-300">
                      {alignment.name}
                    </div>
                  </div>
                  <Stepper
                    value={editable.principle_counts[alignment.code as PrincipleKey]}
                    label={alignment.code}
                    onChange={(value) => onPrincipleChange(alignment.code as PrincipleKey, value)}
                  />
                </div>
              </div>

              {/* Displacement */}
              <div className="p-3 rounded-lg bg-red-50 dark:bg-red-950/30">
                <div className="flex flex-col gap-3 sm:grid sm:grid-cols-[1fr_auto] sm:items-center">
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <span className="font-bold text-red-700 dark:text-red-300">
                        {displacement.code}
                      </span>
                      <button
                        type="button"
                        className="p-1 rounded hover:bg-red-200/50 dark:hover:bg-red-800/50 transition-colors"
                        onClick={() => onPrincipleInfo(displacement.code)}
                        aria-label={`Info about ${displacement.code}`}
                      >
                        <svg className="w-4 h-4 text-red-600 dark:text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                      </button>
                    </div>
                    <div className="text-xs text-red-700 dark:text-red-300">
                      {displacement.name}
                    </div>
                  </div>
                  <Stepper
                    value={editable.principle_counts[displacement.code as PrincipleKey]}
                    label={displacement.code}
                    onChange={(value) => onPrincipleChange(displacement.code as PrincipleKey, value)}
                  />
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Domain Distribution */}
      <div>
        <h3 className="font-semibold text-sm text-gray-500 dark:text-gray-400 uppercase tracking-wide mb-3">
          Distribution by Domain
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left border-b border-gray-200 dark:border-gray-700">
                <th className="pb-2 font-medium">Domain</th>
                <th className="pb-2 font-medium text-center">Displacement</th>
                <th className="pb-2 font-medium text-center">Alignment</th>
              </tr>
            </thead>
            <tbody>
              {(['economy', 'employment', 'education'] as const).map((domain) => {
                const thmDomain = thm.by_domain[domain];
                const gyroDomain = gyroscope.by_domain[domain];
                const thmTotal =
                  thmDomain.GTD + thmDomain.IVD + thmDomain.IAD + thmDomain.IID;
                const gyroTotal =
                  gyroDomain.GMT + gyroDomain.ICV + gyroDomain.IIA + gyroDomain.ICI;
                return (
                  <tr
                    key={domain}
                    className="border-b border-gray-100 dark:border-gray-800"
                  >
                    <td className="py-2 capitalize font-medium">{domain}</td>
                    <td className="py-2 text-center text-red-600 dark:text-red-400">
                      {thmTotal}
                    </td>
                    <td className="py-2 text-center text-emerald-600 dark:text-emerald-400">
                      {gyroTotal}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Compilation Summary */}
      <div className="p-4 rounded-lg bg-gray-50 dark:bg-gray-800/50 space-y-2">
        <h3 className="font-semibold text-sm text-gray-500 dark:text-gray-400 uppercase tracking-wide">
          Compilation
        </h3>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 text-sm">
          <div>
            <span className="text-gray-500">Attestations</span>
            <div className="font-bold text-lg">{compilation.attestation_count}</div>
          </div>
          <div>
            <span className="text-gray-500">Processed</span>
            <div className="font-bold text-lg">{compilation.processed_attestations}</div>
          </div>
          <div>
            <span className="text-gray-500">Bytes</span>
            <div className="font-bold text-lg">{compilation.byte_count}</div>
          </div>
        </div>
      </div>

      {/* Technical Details */}
      <details className="mt-2">
        <summary className="text-sm font-medium cursor-pointer text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200">
          Technical details
        </summary>
        <div className="mt-3 space-y-4">
          {/* Hashes */}
          <div className="p-4 rounded-lg bg-gray-50 dark:bg-gray-800/50 space-y-2">
            <h4 className="font-semibold text-sm text-gray-500 dark:text-gray-400 uppercase tracking-wide">
              Hashes
            </h4>
            <div className="space-y-1 text-xs font-mono break-all">
              <div>
                <span className="text-gray-500">Bytes: </span>
                <span className="text-gray-700 dark:text-gray-300">
                  {compilation.hashes.bytes_sha256}
                </span>
              </div>
              <div>
                <span className="text-gray-500">Events: </span>
                <span className="text-gray-700 dark:text-gray-300">
                  {compilation.hashes.events_sha256}
                </span>
              </div>
            </div>
          </div>

          {/* Warnings */}
          {report.warnings && Object.keys(report.warnings).length > 0 && (
            <div className="p-3 rounded-lg bg-amber-50 dark:bg-amber-950/50 border border-amber-200 dark:border-amber-800">
              <span className="font-semibold text-amber-800 dark:text-amber-200 text-sm">
                Warnings present
              </span>
            </div>
          )}

          {/* Project ID */}
          <div className="text-xs text-gray-500 dark:text-gray-400">
            Project ID: <span className="font-mono">{report.project_id}</span>
          </div>
        </div>
      </details>
    </div>
  );
}
