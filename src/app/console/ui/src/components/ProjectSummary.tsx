import type { ReportCompilation, ReportAccounting } from '../types';

interface ProjectSummaryProps {
  compilation: ReportCompilation | null;
  accounting: ReportAccounting | null;
}

export function ProjectSummaryCard({ compilation, accounting }: ProjectSummaryProps) {
  if (!compilation || !accounting) {
    return null;
  }

  const { thm, gyroscope } = accounting;
  const totalAlignment = gyroscope.totals.GMT + gyroscope.totals.ICV + gyroscope.totals.IIA + gyroscope.totals.ICI;
  const totalDisplacement = thm.totals.GTD + thm.totals.IVD + thm.totals.IAD + thm.totals.IID;
  const totalIncidents = totalAlignment + totalDisplacement;

  return (
    <div className="card p-4 mb-6">
      <h2 className="text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wide mb-4">
        Project Summary
      </h2>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div>
          <div className="text-xs text-gray-500 dark:text-gray-400">Total Incidents</div>
          <div className="text-2xl font-bold">{totalIncidents}</div>
        </div>
        <div>
          <div className="text-xs text-gray-500 dark:text-gray-400">Alignment</div>
          <div className="text-2xl font-bold text-emerald-600 dark:text-emerald-400">{totalAlignment}</div>
        </div>
        <div>
          <div className="text-xs text-gray-500 dark:text-gray-400">Displacement</div>
          <div className="text-2xl font-bold text-red-600 dark:text-red-400">{totalDisplacement}</div>
        </div>
        <div>
          <div className="text-xs text-gray-500 dark:text-gray-400">Attestations</div>
          <div className="text-2xl font-bold">{compilation.processed_attestations}</div>
        </div>
      </div>
    </div>
  );
}

