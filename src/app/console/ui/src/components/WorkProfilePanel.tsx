import type { ReportAccounting } from '../types';

interface WorkProfilePanelProps {
  accounting: ReportAccounting | null;
}

function DistributionBar({ 
  label, 
  data, 
  colors 
}: { 
  label: string; 
  data: Record<string, number>; 
  colors: Record<string, string>;
}) {
  const total = Object.values(data).reduce((a, b) => a + b, 0);
  
  if (total === 0) {
    return (
      <div className="space-y-2">
        <div className="text-sm font-medium text-gray-500">{label}</div>
        <div className="h-4 bg-gray-100 dark:bg-gray-800 rounded-full w-full" />
        <div className="text-xs text-gray-400">No data recorded</div>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex justify-between items-end">
        <span className="text-sm font-medium">{label}</span>
        <span className="text-xs text-gray-500 font-mono">{total} units</span>
      </div>
      
      <div className="h-4 flex rounded-full overflow-hidden">
        {Object.entries(data).map(([key, value]) => {
          if (value === 0) return null;
          const pct = (value / total) * 100;
          return (
            <div
              key={key}
              className={colors[key]}
              style={{ width: `${pct}%` }}
              title={`${key}: ${value} (${pct.toFixed(1)}%)`}
            />
          );
        })}
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
        {Object.entries(data).map(([key, value]) => (
          <div key={key} className="flex items-center gap-2 text-xs">
            <div className={`w-2 h-2 rounded-full ${colors[key]}`} />
            <span className="text-gray-600 dark:text-gray-300 font-medium">
              {key}
            </span>
            <span className="text-gray-400">{value}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export function WorkProfilePanel({ accounting }: WorkProfilePanelProps) {
  if (!accounting) return null;

  const gyroColors: Record<string, string> = {
    GMT: 'bg-indigo-500',
    ICV: 'bg-blue-500',
    IIA: 'bg-cyan-500',
    ICI: 'bg-teal-500',
  };

  const thmColors: Record<string, string> = {
    GTD: 'bg-orange-500',
    IVD: 'bg-red-500',
    IAD: 'bg-pink-500',
    IID: 'bg-purple-500',
  };

  const hasGyro = Object.values(accounting.gyroscope.totals).some(v => v > 0);
  const hasThm = Object.values(accounting.thm.totals).some(v => v > 0);

  if (!hasGyro && !hasThm) return null;

  return (
    <div className="card p-6 space-y-8">
      <h2 className="text-lg font-bold">Work Profile</h2>
      
      <DistributionBar 
        label="Work Composition (Alignment)" 
        data={accounting.gyroscope.totals as unknown as Record<string, number>} 
        colors={gyroColors} 
      />

      <div className="border-t border-gray-100 dark:border-gray-800 pt-6">
        <DistributionBar 
          label="Risk Coverage (Displacement)" 
          data={accounting.thm.totals as unknown as Record<string, number>} 
          colors={thmColors} 
        />
      </div>
    </div>
  );
}


