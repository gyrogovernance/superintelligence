import { Stepper } from './Stepper';
import type { EditableState, PrincipleKey } from '../types';

interface PrinciplesPanelProps {
  counts: EditableState['principle_counts'];
  onChange: (principle: PrincipleKey, value: number) => void;
  onInfo: (principle: string) => void;
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

export function PrinciplesPanel({ counts, onChange, onInfo }: PrinciplesPanelProps) {
  return (
    <div className="space-y-4">
      {STAGES.map(({ key, title, alignment, displacement }) => (
        <div key={key} className="card p-4 space-y-3">
          <h3 className="font-semibold text-sm text-gray-500 dark:text-gray-400 uppercase tracking-wide">
            {title}
          </h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {/* Alignment */}
            <div className="p-3 rounded-lg bg-emerald-50 dark:bg-emerald-950/30 space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="font-bold text-emerald-700 dark:text-emerald-300">
                    {alignment.code}
                  </span>
                  <button
                    type="button"
                    className="p-1 rounded hover:bg-emerald-200/50 dark:hover:bg-emerald-800/50 transition-colors"
                    onClick={() => onInfo(alignment.code)}
                    aria-label={`Info about ${alignment.code}`}
                  >
                    <svg className="w-4 h-4 text-emerald-600 dark:text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  </button>
                </div>
                <Stepper
                  value={counts[alignment.code as PrincipleKey]}
                  label={alignment.code}
                  onChange={(value) => onChange(alignment.code as PrincipleKey, value)}
                />
              </div>
              <div className="text-xs text-emerald-700 dark:text-emerald-300 truncate">
                {alignment.name}
              </div>
            </div>

            {/* Displacement */}
            <div className="p-3 rounded-lg bg-red-50 dark:bg-red-950/30 space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="font-bold text-red-700 dark:text-red-300">
                    {displacement.code}
                  </span>
                  <button
                    type="button"
                    className="p-1 rounded hover:bg-red-200/50 dark:hover:bg-red-800/50 transition-colors"
                    onClick={() => onInfo(displacement.code)}
                    aria-label={`Info about ${displacement.code}`}
                  >
                    <svg className="w-4 h-4 text-red-600 dark:text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  </button>
                </div>
                <Stepper
                  value={counts[displacement.code as PrincipleKey]}
                  label={displacement.code}
                  onChange={(value) => onChange(displacement.code as PrincipleKey, value)}
                />
              </div>
              <div className="text-xs text-red-700 dark:text-red-300 truncate">
                {displacement.name}
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
