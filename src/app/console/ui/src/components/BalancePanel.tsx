import type { ReportApertures } from '../types';

interface BalancePanelProps {
  apertures: ReportApertures | null;
  aStar: number;
}

function computeSI(aperture: number, aStar: number): number {
  if (aperture <= 0) return 0;
  const ratio = aperture / aStar;
  const inverseRatio = aStar / aperture;
  return 100 / Math.max(ratio, inverseRatio);
}

function formatSI(si: number, aperture: number, aStar: number) {
  const text = si.toFixed(1);
  let status: 'balanced' | 'warning' | 'critical';
  if (si >= 90) {
    status = 'balanced';
  } else if (si >= 70) {
    status = 'warning';
  } else {
    status = 'critical';
  }
  
  let direction: 'rigid' | 'fragmented' | 'balanced';
  const deviationPercent = Math.abs((aperture - aStar) / aStar) * 100;
  if (deviationPercent <= 1) {
    direction = 'balanced';
  } else if (aperture < aStar) {
    direction = 'rigid';
  } else {
    direction = 'fragmented';
  }
  
  return { text, status, direction, si };
}

interface GaugeProps {
  label: string;
  aperture: number;
  aStar: number;
}

function Gauge({ label, aperture, aStar }: GaugeProps) {
  const si = computeSI(aperture, aStar);
  const { text, status, direction } = formatSI(si, aperture, aStar);

  const statusColors = {
    balanced: 'bg-balance-balanced',
    warning: 'bg-balance-warning',
    critical: 'bg-balance-critical',
  };

  const directionColors: Record<'rigid' | 'fragmented' | 'balanced', string> = {
    rigid: 'bg-balance-rigid',
    fragmented: 'bg-balance-fragmented',
    balanced: 'bg-balance-balanced',
  };

  const fillColor = status === 'balanced' ? statusColors.balanced : directionColors[direction as keyof typeof directionColors];
  
  const deviationFromOptimal = 100 - si;
  const fillPercent = Math.min(50, deviationFromOptimal / 2);

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="font-medium">{label}</span>
        <div className="flex items-center gap-2">
          <span className="sr-only">Aperture {aperture.toFixed(4)}</span>
          <span
            className={`badge ${
              status === 'balanced'
                ? 'badge-success'
                : status === 'warning'
                ? 'badge-warning'
                : 'badge-danger'
            }`}
            title={`Aperture: ${aperture.toFixed(4)} | SI: ${si.toFixed(1)}`}
          >
            SI: {text}
          </span>
        </div>
      </div>
      <div className="gauge-track">
        <div className="gauge-centre" />
        {direction === 'rigid' ? (
          <div
            className={`gauge-fill ${fillColor}`}
            style={{
              right: '50%',
              width: `${fillPercent}%`,
            }}
          />
        ) : direction === 'fragmented' ? (
          <div
            className={`gauge-fill ${fillColor}`}
            style={{
              left: '50%',
              width: `${fillPercent}%`,
            }}
          />
        ) : (
          <div
            className={`gauge-fill ${fillColor}`}
            style={{
              left: '50%',
              width: '2px',
              transform: 'translateX(-50%)',
            }}
          />
        )}
      </div>
      <div className="flex justify-between text-xs text-gray-500">
        <span>Rigid</span>
        <span>A* = {aStar.toFixed(4)}</span>
        <span>Fragmented</span>
      </div>
    </div>
  );
}

export function BalancePanel({ apertures, aStar }: BalancePanelProps) {
  if (!apertures) {
    return (
      <div className="card p-6">
        <h2 className="text-lg font-bold mb-4">Balance</h2>
        <p className="text-gray-500 dark:text-gray-400">
          No aperture data available. Sync a project to see balance metrics.
        </p>
      </div>
    );
  }

  return (
    <div className="card p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-bold">Balance</h2>
        <span className="text-sm text-gray-500 dark:text-gray-400">
          Target A* = {aStar.toFixed(4)}
        </span>
      </div>
      <Gauge label="Economy" aperture={apertures.A_econ} aStar={aStar} />
      <Gauge label="Employment" aperture={apertures.A_emp} aStar={aStar} />
      <Gauge label="Education" aperture={apertures.A_edu} aStar={aStar} />
    </div>
  );
}
