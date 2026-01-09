import type { ReportCompilation, ReportAccounting, EditableState } from '../types';

interface ProgramSummaryProps {
  compilation: ReportCompilation | null;
  accounting: ReportAccounting | null;
  unit: EditableState['unit'];
  lastSynced: string | null;
  domainCounts?: EditableState['domain_counts'];
}

export function ProgramSummaryCard({ compilation, accounting, unit, lastSynced, domainCounts }: ProgramSummaryProps) {
  if (!compilation || !accounting) {
    return null;
  }

  const { thm, gyroscope } = accounting;
  const totalAlignment = gyroscope.totals.GMT + gyroscope.totals.ICV + gyroscope.totals.IIA + gyroscope.totals.ICI;
  const totalDisplacement = thm.totals.GTD + thm.totals.IVD + thm.totals.IAD + thm.totals.IID;
  const totalIncidents = totalAlignment + totalDisplacement;
  
  const QUALIFICATION_THRESHOLD = 5;
  // Qualification based on alignment work only (not displacement risk signatures)
  const isSprintQualified = unit === 'sprint' || totalAlignment >= QUALIFICATION_THRESHOLD;
  
  const UNIT_RATE = unit === 'daily' ? 120 : 480;
  // Only alignment units (principles) are billable; displacement is risk coverage, not work
  const estimatedValue = totalAlignment * UNIT_RATE;

  // Calculate domain imbalance warnings for badge
  const hasDomainImbalance = domainCounts ? (() => {
    const total = domainCounts.economy + domainCounts.employment + domainCounts.education;
    if (total === 0) return false; // Will be evenly distributed, no imbalance warning
    const econPct = domainCounts.economy / total;
    const empPct = domainCounts.employment / total;
    const eduPct = domainCounts.education / total;
    // Check for missing domains or single-domain dominance (>80%)
    return domainCounts.economy === 0 || domainCounts.employment === 0 || domainCounts.education === 0 ||
           econPct > 0.8 || empPct > 0.8 || eduPct > 0.8;
  })() : false;

  return (
    <div className="card p-6 mb-6">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-6">
        <div>
          <h2 className="text-lg font-bold">Program Status</h2>
          <div className="flex items-center gap-2 mt-1 flex-wrap">
            <span className={`badge ${unit === 'daily' ? 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200' : 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200'}`}>
              {unit === 'daily' ? 'Daily Prize Track' : 'Sprint Stipend Track'}
            </span>
            {unit === 'daily' && isSprintQualified && (
              <span className="badge bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200">
                Sprint Qualified
              </span>
            )}
            {hasDomainImbalance && (
              <span className="badge bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-200">
                Domain Imbalance
              </span>
            )}
          </div>
        </div>
        <div className="mt-4 sm:mt-0 text-right">
           <div className="text-xs text-gray-500 dark:text-gray-400">Estimated Value</div>
           <div className="text-2xl font-bold font-mono">£{estimatedValue.toLocaleString()}</div>
           <div className="text-xs text-gray-400">@ £{UNIT_RATE}/{unit} unit</div>
           {lastSynced && (
             <div className="mt-2 text-xs text-gray-400">
               Last synced: {new Date(lastSynced).toLocaleDateString()}
             </div>
           )}
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-6 pt-6 border-t border-gray-100 dark:border-gray-800">
        <div>
          <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">Safety Coverage</div>
          <div className="text-2xl font-bold">{totalIncidents}</div>
          <div className="text-xs text-gray-400">Total Incidents</div>
          {unit === 'daily' && !isSprintQualified && (
            <div className="text-xs text-amber-600 dark:text-amber-400 mt-1">
              {QUALIFICATION_THRESHOLD - totalAlignment} more alignment units to qualify
            </div>
          )}
        </div>
        
        <div>
          <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">Work Type</div>
          <div className="text-2xl font-bold text-indigo-600 dark:text-indigo-400">
            {totalAlignment}
          </div>
          <div className="text-xs text-gray-400">Alignment Units</div>
        </div>

        <div>
          <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">Risk Coverage</div>
          <div className="text-2xl font-bold text-orange-600 dark:text-orange-400">
            {totalDisplacement}
          </div>
          <div className="text-xs text-gray-400">Displacement Units</div>
        </div>

        <div>
          <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">Verification</div>
          <div className="text-2xl font-bold font-mono text-emerald-600 dark:text-emerald-400">
             {compilation.processed_attestations}
          </div>
          <div className="text-xs text-gray-400">Total Attestations</div>
        </div>
      </div>
    </div>
  );
}
