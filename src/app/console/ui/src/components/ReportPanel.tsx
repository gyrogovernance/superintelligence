import { useState, useEffect } from 'react';
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
  onDomainInfo: (domain: DomainKey) => void;
  onPrincipleInfo: (principle: string) => void;
  onAgentsChange: (value: string) => void;
  onAgenciesChange: (value: string) => void;
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
  { key: 'economy', label: 'Economy (Kernel)', sublabel: 'Structural substrate / state' },
  { key: 'employment', label: 'Employment (Gyroscope)', sublabel: 'Active work / principles' },
  { key: 'education', label: 'Education (THM)', sublabel: 'Measurements / displacements' },
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
  onAgentsChange,
  onAgenciesChange,
  hasEventLog = false,
}: ReportPanelProps) {
  // Helper function to clean placeholder text
  const cleanPlaceholder = (value: string, placeholder: string) => {
    const cleaned = value || '';
    return cleaned === placeholder || cleaned.trim() === placeholder ? '' : cleaned;
  };

  // Local state for text fields to avoid controlled input issues
  const [localNotes, setLocalNotes] = useState(notes);
  const [localAgents, setLocalAgents] = useState(() => {
    const agentsValue = editable?.agents || '';
    return cleanPlaceholder(agentsValue, "(Names of people involved in this program)");
  });
  const [localAgencies, setLocalAgencies] = useState(() => {
    const agenciesValue = editable?.agencies || '';
    return cleanPlaceholder(agenciesValue, "(Names of agencies involved in this program)");
  });

  // Sync local state when props change (e.g., program switch)
  useEffect(() => {
    setLocalNotes(notes);
  }, [notes]);

  useEffect(() => {
    // Clean up placeholder text if it somehow got into the value
    const agentsValue = editable?.agents || '';
    const cleaned = cleanPlaceholder(agentsValue, "(Names of people involved in this program)");
    setLocalAgents(cleaned);
  }, [editable?.agents]);

  useEffect(() => {
    // Clean up placeholder text if it somehow got into the value
    const agenciesValue = editable?.agencies || '';
    const cleaned = cleanPlaceholder(agenciesValue, "(Names of agencies involved in this program)");
    setLocalAgencies(cleaned);
  }, [editable?.agencies]);

  if (!report || !editable) {
    return (
      <div className="card p-6">
        <h2 className="text-lg font-bold mb-2">Report</h2>
        <p className="text-gray-500 dark:text-gray-400">
          No report data. Create or sync a program to generate a report.
        </p>
      </div>
    );
  }

  const { accounting, compilation } = report;
  const { thm, gyroscope } = accounting;
  const domainTotal = editable.domain_counts.economy + editable.domain_counts.employment + editable.domain_counts.education;
  
  // Calculate domain imbalance warnings
  const domainImbalanceWarnings: string[] = [];
  if (domainTotal > 0) {
    const econPct = editable.domain_counts.economy / domainTotal;
    const empPct = editable.domain_counts.employment / domainTotal;
    const eduPct = editable.domain_counts.education / domainTotal;
    
    // Check for missing domains
    if (editable.domain_counts.economy === 0) {
      domainImbalanceWarnings.push("No Economy coverage declared. GGG recommends considering system-level structures and flows.");
    }
    if (editable.domain_counts.employment === 0) {
      domainImbalanceWarnings.push("No Employment coverage declared. GGG recommends considering impacts on work and roles.");
    }
    if (editable.domain_counts.education === 0) {
      domainImbalanceWarnings.push("No Education coverage declared. GGG recommends considering capacities and epistemic literacy.");
    }
    
    // Check for single-domain dominance (>80%)
    if (econPct > 0.8) {
      domainImbalanceWarnings.push(">80% of coverage is in Economy. This may indicate ungrounded governance if Employment and Education impacts are not also considered.");
    }
    if (empPct > 0.8) {
      domainImbalanceWarnings.push(">80% of coverage is in Employment. Consider system-level (Economy) and capacity (Education) dimensions.");
    }
    if (eduPct > 0.8) {
      domainImbalanceWarnings.push(">80% of coverage is in Education. Consider structural (Economy) and work (Employment) dimensions.");
    }
  } else {
    domainImbalanceWarnings.push("All domain counts are zero. Incidents will be distributed evenly across domains.");
  }

  return (
    <div className="card p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-bold">Report</h2>
        <div className="flex gap-2">
          <button
            type="button"
            className="btn btn-secondary btn-sm"
            onClick={() => {
              const { accounting } = report;
              const { thm, gyroscope } = accounting;
              const totalAlignment = gyroscope.totals.GMT + gyroscope.totals.ICV + gyroscope.totals.IIA + gyroscope.totals.ICI;
              const totalDisplacement = thm.totals.GTD + thm.totals.IVD + thm.totals.IAD + thm.totals.IID;
              const totalIncidents = totalAlignment + totalDisplacement;
              const alignmentRatio = totalIncidents > 0 ? (totalAlignment / totalIncidents) * 100 : 0;

              let participantsSection = '';
              if (editable.agents || editable.agencies) {
                participantsSection = '\n## Participants\n\n';
                if (editable.agents) {
                  participantsSection += `### Agents\n\n${editable.agents}\n\n`;
                }
                if (editable.agencies) {
                  participantsSection += `### Agencies\n\n${editable.agencies}\n\n`;
                }
              }

              const receipt = `# Work Profile Receipt
Program: ${report.program_slug}
Program ID: ${report.program_id}
Date: ${new Date().toISOString().split('T')[0]}
${participantsSection}
## Common Source Consensus

All Artificial categories of Authority and Agency are Derivatives
originating from Human Intelligence.

## Summary
Total Incidents: ${totalIncidents}
Alignment Ratio: ${alignmentRatio.toFixed(1)}%
Alignment Work: ${totalAlignment} units
Displacement Work: ${totalDisplacement} units

## Work Composition (Alignment)
- GMT (Governance Management Traceability): ${gyroscope.totals.GMT}
- ICV (Information Curation Variety): ${gyroscope.totals.ICV}
- IIA (Inference Interaction Accountability): ${gyroscope.totals.IIA}
- ICI (Intelligence Cooperation Integrity): ${gyroscope.totals.ICI}

## Risk Coverage (Displacement)
- GTD (Governance Traceability Displacement): ${thm.totals.GTD}
- IVD (Information Variety Displacement): ${thm.totals.IVD}
- IAD (Inference Accountability Displacement): ${thm.totals.IAD}
- IID (Intelligence Integrity Displacement): ${thm.totals.IID}

## Compilation
Attestations Processed: ${compilation.processed_attestations}
Total Attestations: ${compilation.attestation_count}
Bytes Processed: ${compilation.byte_count}

## Verification Hashes
Bytes SHA256: ${compilation.hashes.bytes_sha256}
Events SHA256: ${compilation.hashes.events_sha256}

---
Generated by AIR Console
`;

              const blob = new Blob([receipt], { type: 'text/markdown' });
              const url = URL.createObjectURL(blob);
              const a = document.createElement('a');
              a.href = url;
              a.download = `${report.program_slug}-receipt-${new Date().toISOString().split('T')[0]}.md`;
              document.body.appendChild(a);
              a.click();
              document.body.removeChild(a);
              URL.revokeObjectURL(url);
            }}
          >
            Download Receipt
          </button>
        <button type="button" className="btn btn-primary btn-sm" onClick={onDownloadBundle}>
          Download Bundle
        </button>
        </div>
      </div>

      {/* Participants */}
      <div className="space-y-4">
        <h3 className="font-semibold text-sm text-gray-500 dark:text-gray-400 uppercase tracking-wide">
          Participants
        </h3>
        
        <div className="space-y-2">
          <label htmlFor="agents-input" className="block text-sm font-medium">
            Agents
          </label>
          <textarea
            id="agents-input"
            className="input w-full min-h-[60px] resize-y"
            placeholder="Names of people involved in this program..."
            value={localAgents}
            onChange={(e) => setLocalAgents(e.target.value)}
            onBlur={() => onAgentsChange(localAgents)}
          />
        </div>

        <div className="space-y-2">
          <label htmlFor="agencies-input" className="block text-sm font-medium">
            Agencies
          </label>
          <textarea
            id="agencies-input"
            className="input w-full min-h-[60px] resize-y"
            placeholder="Names of agencies involved in this program..."
            value={localAgencies}
            onChange={(e) => setLocalAgencies(e.target.value)}
            onBlur={() => onAgenciesChange(localAgencies)}
          />
        </div>
      </div>

      {/* Notes */}
      <div className="space-y-2">
        <h3 className="font-semibold text-sm text-gray-500 dark:text-gray-400 uppercase tracking-wide">
          Notes
        </h3>
        <textarea
          id="notes-input"
          className="input w-full min-h-[80px] resize-y"
          placeholder="Add context or key observations for this program..."
          value={localNotes}
          onChange={(e) => setLocalNotes(e.target.value)}
          onBlur={() => onNotesChange(localNotes)}
        />
        <p className="text-xs text-gray-400 dark:text-gray-500">
          Notes are for human context and do not affect the compiled ledger or apertures.
        </p>
      </div>

      {/* Domains */}
      <div className="space-y-4">
        <div>
          <h3 className="font-semibold text-sm text-gray-500 dark:text-gray-400 uppercase tracking-wide mb-2">
            Domains
          </h3>
          {hasEventLog ? (
            <div className="mb-3 p-2 rounded bg-blue-50 dark:bg-blue-950/30 border border-blue-200 dark:border-blue-800">
              <p className="text-sm text-blue-800 dark:text-blue-200">
                <strong>Real Mode:</strong> Domain counts are auto-generated from the event log. 
                Per GGG hierarchy: Economy = Kernel (structural substrate), Employment = Gyroscope (active work), Education = THM (measurements).
              </p>
            </div>
          ) : (
            <>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                <strong>Simulation Mode:</strong> Domain counts are manually set to simulate domain distribution. 
                Adjust these to simulate how incidents are attributed across domains.
              </p>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                Per GGG hierarchy:
              </p>
              <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1 mb-4 ml-4 list-disc">
                <li><strong>Economy (Kernel):</strong> Structural substrate / state (CGM operations)</li>
                <li><strong>Employment (Gyroscope):</strong> Active work / principles (alignment labor)</li>
                <li><strong>Education (THM):</strong> Measurements / displacements (risk signatures)</li>
              </ul>
            </>
          )}
        </div>
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
                    disabled={hasEventLog}
                  />
                </div>
                <div className="text-xs text-gray-500 dark:text-gray-400">{sublabel}</div>
              </div>
            );
          })}
        </div>
        <div className="pt-2 border-t border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between text-sm mb-3">
            <span className="text-gray-500">Total Domain Count</span>
            <span className="font-bold">{domainTotal}</span>
          </div>
          
          {/* Domain Imbalance Warnings */}
          {domainImbalanceWarnings.length > 0 && (
            <div className="mt-3 p-3 rounded-lg bg-amber-50 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-800">
              <div className="flex items-start gap-2">
                <svg className="w-5 h-5 text-amber-600 dark:text-amber-400 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
                <div className="flex-1">
                  <div className="text-sm font-medium text-amber-800 dark:text-amber-200 mb-1">
                    Domain Coverage Warning
                  </div>
                  <ul className="text-xs text-amber-700 dark:text-amber-300 space-y-1 list-disc list-inside">
                    {domainImbalanceWarnings.map((warning, idx) => (
                      <li key={idx}>{warning}</li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* GGG Stages */}
      <div className="space-y-4">
        <h3 className="font-semibold text-sm text-gray-500 dark:text-gray-400 uppercase tracking-wide">
          Principles by Stage
        </h3>
        {STAGES.map(({ key, title, alignment, displacement }) => {
          const alignValue = editable.principle_counts[alignment.code as PrincipleKey];
          const dispValue = editable.principle_counts[displacement.code as PrincipleKey];
          const total = alignValue + dispValue;
          const alignPct = total > 0 ? (alignValue / total) * 100 : 50;
          const dispPct = total > 0 ? (dispValue / total) * 100 : 50;

          return (
          <div key={key} className="p-4 rounded-lg bg-gray-50 dark:bg-gray-800/50 space-y-3">
            <h4 className="font-semibold text-sm text-gray-700 dark:text-gray-300">{title}</h4>
              
              {/* Tension Bar */}
              <div className="space-y-2">
                <div className="h-8 flex rounded-lg overflow-hidden border border-gray-200 dark:border-gray-700">
                  <div
                    className="bg-emerald-500 flex items-center justify-end pr-2 transition-all"
                    style={{ width: `${alignPct}%` }}
                  >
                    {alignPct > 15 && (
                      <span className="text-white text-xs font-semibold">{alignment.code}</span>
                    )}
                  </div>
                  <div
                    className="bg-red-500 flex items-center justify-start pl-2 transition-all"
                    style={{ width: `${dispPct}%` }}
                  >
                    {dispPct > 15 && (
                      <span className="text-white text-xs font-semibold">{displacement.code}</span>
                    )}
                  </div>
                </div>
                <div className="flex justify-between text-xs text-gray-500">
                  <span>{alignValue} {alignment.code}</span>
                  <span>{dispValue} {displacement.code}</span>
                </div>
              </div>

              {/* Controls */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div className="p-3 rounded-lg bg-emerald-50 dark:bg-emerald-950/30">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="font-bold text-emerald-700 dark:text-emerald-300 text-sm">
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
                  <div className="text-xs text-emerald-700 dark:text-emerald-300 mb-2">
                      {alignment.name}
                  </div>
                  <Stepper
                    value={alignValue}
                    label={alignment.code}
                    onChange={(value) => onPrincipleChange(alignment.code as PrincipleKey, value)}
                  />
              </div>

              <div className="p-3 rounded-lg bg-red-50 dark:bg-red-950/30">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="font-bold text-red-700 dark:text-red-300 text-sm">
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
                  <div className="text-xs text-red-700 dark:text-red-300 mb-2">
                      {displacement.name}
                  </div>
                  <Stepper
                    value={dispValue}
                    label={displacement.code}
                    onChange={(value) => onPrincipleChange(displacement.code as PrincipleKey, value)}
                  />
                </div>
              </div>
            </div>
          );
        })}
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

          {/* Program ID */}
          <div className="text-xs text-gray-500 dark:text-gray-400">
            Program ID: <span className="font-mono">{report.program_id}</span>
          </div>
        </div>
      </details>
    </div>
  );
}
