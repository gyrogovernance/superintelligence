import { Stepper } from './Stepper';
import type { EditableState, DomainKey } from '../types';

interface DomainsPanelProps {
  counts: EditableState['domain_counts'];
  onChange: (domain: DomainKey, value: number) => void;
  onInfo: (domain: string) => void;
}

export function DomainsPanel({ counts, onChange, onInfo }: DomainsPanelProps) {
  const total = counts.economy + counts.employment + counts.education;

  const domains: Array<{
    key: DomainKey;
    label: string;
    sublabel: string;
    color: string;
  }> = [
    { key: 'economy', label: 'Economy', sublabel: 'CGM operations', color: 'bg-blue-500' },
    { key: 'employment', label: 'Employment', sublabel: 'Gyroscope work', color: 'bg-purple-500' },
    { key: 'education', label: 'Education', sublabel: 'THM capacities', color: 'bg-amber-500' },
  ];

  return (
    <div className="card p-6 space-y-4">
      <h2 className="text-lg font-bold">Domains</h2>
      <div className="space-y-4">
        {domains.map(({ key, label, sublabel, color }) => {
          const percentage = total > 0 ? ((counts[key] / total) * 100).toFixed(0) : '0';
          return (
            <div key={key} className="space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="font-medium">{label}</span>
                  <button
                    type="button"
                    className="p-1 rounded hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
                    onClick={() => onInfo(key)}
                    aria-label={`Info about ${label}`}
                  >
                    <svg className="w-4 h-4 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  </button>
                </div>
                <Stepper
                  value={counts[key]}
                  label={label}
                  onChange={(value) => onChange(key, value)}
                />
              </div>
              <div className="text-xs text-gray-500 dark:text-gray-400">{sublabel}</div>
              <div className="h-2 rounded-full bg-gray-200 dark:bg-gray-700 overflow-hidden">
                <div
                  className={`h-full ${color} transition-all duration-300`}
                  style={{ width: `${percentage}%` }}
                />
              </div>
              <div className="text-xs text-right text-gray-500">{percentage}%</div>
            </div>
          );
        })}
      </div>
      <div className="pt-2 border-t border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between text-sm">
          <span className="text-gray-500">Total Domain Count</span>
          <span className="font-bold">{total}</span>
        </div>
      </div>
    </div>
  );
}

