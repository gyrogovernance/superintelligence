import { useState, useEffect, useMemo } from 'react';
import type { ProgramSummary, ReportAccounting, DomainKey } from '../types';
import * as api from '../api';
import { WorkProfilePanel } from './WorkProfilePanel';

interface PortfolioViewProps {
  programs: ProgramSummary[];
  onSelectProgram: (slug: string) => void;
}

interface ProgramData {
  slug: string;
  accounting: ReportAccounting;
  unit: 'daily' | 'sprint';
  lastSynced: string | null;
  domainCounts?: { economy: number; employment: number; education: number };
  signed?: boolean;
  verified?: boolean;
  hasEventLog?: boolean;
  ecology?: {
    total_capacity_MU: number;
    used_capacity_MU: number;
    free_capacity_MU: number;
  };
}

function aggregateAccounting(data: ProgramData[]): ReportAccounting {
  const aggregated: ReportAccounting = {
    gyroscope: {
      totals: { GMT: 0, ICV: 0, IIA: 0, ICI: 0 },
      by_domain: {
        economy: { GMT: 0, ICV: 0, IIA: 0, ICI: 0 },
        employment: { GMT: 0, ICV: 0, IIA: 0, ICI: 0 },
        education: { GMT: 0, ICV: 0, IIA: 0, ICI: 0 },
      },
    },
    thm: {
      totals: { GTD: 0, IVD: 0, IAD: 0, IID: 0 },
      by_domain: {
        economy: { GTD: 0, IVD: 0, IAD: 0, IID: 0 },
        employment: { GTD: 0, IVD: 0, IAD: 0, IID: 0 },
        education: { GTD: 0, IVD: 0, IAD: 0, IID: 0 },
      },
    },
  };

  for (const item of data) {
    const acc = item.accounting;
    aggregated.gyroscope.totals.GMT += acc.gyroscope.totals.GMT;
    aggregated.gyroscope.totals.ICV += acc.gyroscope.totals.ICV;
    aggregated.gyroscope.totals.IIA += acc.gyroscope.totals.IIA;
    aggregated.gyroscope.totals.ICI += acc.gyroscope.totals.ICI;

    aggregated.thm.totals.GTD += acc.thm.totals.GTD;
    aggregated.thm.totals.IVD += acc.thm.totals.IVD;
    aggregated.thm.totals.IAD += acc.thm.totals.IAD;
    aggregated.thm.totals.IID += acc.thm.totals.IID;

    for (const domain of ['economy', 'employment', 'education'] as const) {
      aggregated.gyroscope.by_domain[domain].GMT += acc.gyroscope.by_domain[domain].GMT;
      aggregated.gyroscope.by_domain[domain].ICV += acc.gyroscope.by_domain[domain].ICV;
      aggregated.gyroscope.by_domain[domain].IIA += acc.gyroscope.by_domain[domain].IIA;
      aggregated.gyroscope.by_domain[domain].ICI += acc.gyroscope.by_domain[domain].ICI;

      aggregated.thm.by_domain[domain].GTD += acc.thm.by_domain[domain].GTD;
      aggregated.thm.by_domain[domain].IVD += acc.thm.by_domain[domain].IVD;
      aggregated.thm.by_domain[domain].IAD += acc.thm.by_domain[domain].IAD;
      aggregated.thm.by_domain[domain].IID += acc.thm.by_domain[domain].IID;
    }
  }

  return aggregated;
}

export function PortfolioView({ programs, onSelectProgram }: PortfolioViewProps) {
  const [programData, setProgramData] = useState<ProgramData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [domainFilter, setDomainFilter] = useState<DomainKey | 'all'>('all');
  const [highDisplacementFilter, setHighDisplacementFilter] = useState(false);
  const [multiDomainFilter, setMultiDomainFilter] = useState(false);
  const [onlyVerifiedFilter, setOnlyVerifiedFilter] = useState(false);
  const [onlySignedFilter, setOnlySignedFilter] = useState(false);
  const [onlyRealModeFilter, setOnlyRealModeFilter] = useState(false);
  const [sortColumn, setSortColumn] = useState<'slug' | 'unit' | 'incidents' | 'ratio' | 'value'>('slug');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc');
  const [annualCapacity, setAnnualCapacity] = useState<number | null>(null);

  useEffect(() => {
    async function loadAnnualCapacity() {
      try {
        const res = await fetch('/api/capacity/annual');
        if (res.ok) {
          const data = await res.json();
          setAnnualCapacity(data.annual_capacity_MU);
        }
      } catch {
        // Ignore errors, annual capacity is optional
      }
    }
    loadAnnualCapacity();
  }, []);

  useEffect(() => {
    async function loadAllReports() {
      setLoading(true);
      setError(null);
      try {
        const reports = await Promise.all(
          programs.map(async (p) => {
            try {
              const program = await api.getProgram(p.slug);
              if (program.report?.accounting && program.editable) {
                return { 
                  slug: p.slug, 
                  accounting: program.report.accounting,
                  unit: program.editable.unit,
                  lastSynced: program.last_synced,
                  domainCounts: program.editable.domain_counts,
                  signed: program.governance?.signed || false,
                  verified: program.governance?.verified || false,
                  hasEventLog: program.has_event_log || false,
                  ecology: program.ecology ? {
                    total_capacity_MU: program.ecology.total_capacity_MU,
                    used_capacity_MU: program.ecology.used_capacity_MU,
                    free_capacity_MU: program.ecology.free_capacity_MU,
                  } : undefined,
                };
              }
              return null;
            } catch {
              return null;
            }
          })
        );
        const validPrograms = reports.filter((p): p is ProgramData => p !== null);
        setProgramData(validPrograms);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load portfolio data');
      } finally {
        setLoading(false);
      }
    }

    if (programs.length > 0) {
      loadAllReports();
    } else {
      setProgramData([]);
      setLoading(false);
    }
  }, [programs]);

  const filteredPrograms = useMemo(() => {
    let filtered = programData;

    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter((p) => p.slug.toLowerCase().includes(query));
    }

    if (domainFilter !== 'all') {
      filtered = filtered.filter((p) => {
        const domainData = p.accounting.gyroscope.by_domain[domainFilter];
        const domainTotal = domainData.GMT + domainData.ICV + domainData.IIA + domainData.ICI;
        return domainTotal > 0;
      });
    }

    if (highDisplacementFilter) {
      filtered = filtered.filter((p) => {
        const totalDisplacement = p.accounting.thm.totals.GTD + p.accounting.thm.totals.IVD +
                                  p.accounting.thm.totals.IAD + p.accounting.thm.totals.IID;
        const totalAlignment = p.accounting.gyroscope.totals.GMT + p.accounting.gyroscope.totals.ICV +
                               p.accounting.gyroscope.totals.IIA + p.accounting.gyroscope.totals.ICI;
        const total = totalDisplacement + totalAlignment;
        const displacementRatio = total > 0 ? totalDisplacement / total : 0;
        return displacementRatio > 0.5;
      });
    }

    if (multiDomainFilter) {
      filtered = filtered.filter((p) => {
        if (!p.domainCounts) return false;
        // Multi-domain: all three domains must have count > 0
        return p.domainCounts.economy > 0 && p.domainCounts.employment > 0 && p.domainCounts.education > 0;
      });
    }

    if (onlyVerifiedFilter) {
      filtered = filtered.filter((p) => p.verified === true);
    }

    if (onlySignedFilter) {
      filtered = filtered.filter((p) => p.signed === true);
    }

    if (onlyRealModeFilter) {
      filtered = filtered.filter((p) => p.hasEventLog === true);
    }

    return filtered;
  }, [programData, searchQuery, domainFilter, highDisplacementFilter, multiDomainFilter, onlyVerifiedFilter, onlySignedFilter, onlyRealModeFilter]);

  const sortedPrograms = useMemo(() => {
    const sorted = [...filteredPrograms].sort((a, b) => {
      let aVal: number | string = 0;
      let bVal: number | string = 0;

      if (sortColumn === 'slug') {
        aVal = a.slug;
        bVal = b.slug;
      } else if (sortColumn === 'unit') {
        aVal = a.unit;
        bVal = b.unit;
      } else {
        const alignA = a.accounting.gyroscope.totals;
        const dispA = a.accounting.thm.totals;
        const totalAlignA = alignA.GMT + alignA.ICV + alignA.IIA + alignA.ICI;
        const totalDispA = dispA.GTD + dispA.IVD + dispA.IAD + dispA.IID;
        const totalA = totalAlignA + totalDispA;
        const rateA = a.unit === 'daily' ? 120 : 480;
        // Only alignment units are billable; displacement is risk coverage
        const valA = totalAlignA * rateA;
        const ratioA = totalA > 0 ? (totalAlignA / totalA) * 100 : 0;

        const alignB = b.accounting.gyroscope.totals;
        const dispB = b.accounting.thm.totals;
        const totalAlignB = alignB.GMT + alignB.ICV + alignB.IIA + alignB.ICI;
        const totalDispB = dispB.GTD + dispB.IVD + dispB.IAD + dispB.IID;
        const totalB = totalAlignB + totalDispB;
        const rateB = b.unit === 'daily' ? 120 : 480;
        // Only alignment units are billable; displacement is risk coverage
        const valB = totalAlignB * rateB;
        const ratioB = totalB > 0 ? (totalAlignB / totalB) * 100 : 0;

        if (sortColumn === 'incidents') {
          aVal = totalA;
          bVal = totalB;
        } else if (sortColumn === 'ratio') {
          aVal = ratioA;
          bVal = ratioB;
        } else if (sortColumn === 'value') {
          aVal = valA;
          bVal = valB;
        }
      }

      if (typeof aVal === 'string' && typeof bVal === 'string') {
        return sortDirection === 'asc' ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
      } else {
        return sortDirection === 'asc' ? (aVal as number) - (bVal as number) : (bVal as number) - (aVal as number);
      }
    });
    return sorted;
  }, [filteredPrograms, sortColumn, sortDirection]);

  const handleSort = (column: typeof sortColumn) => {
    if (sortColumn === column) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortColumn(column);
      setSortDirection('asc');
    }
  };

  const aggregated = useMemo(() => {
    return aggregateAccounting(filteredPrograms);
  }, [filteredPrograms]);

  const totalValue = useMemo(() => {
    return filteredPrograms.reduce((sum, p) => {
      const align = p.accounting.gyroscope.totals;
      const totalAlignment = align.GMT + align.ICV + align.IIA + align.ICI;
      
      const rate = p.unit === 'daily' ? 120 : 480;
      // Only alignment units (principles) are billable; displacement is risk coverage
      return sum + (totalAlignment * rate);
    }, 0);
  }, [filteredPrograms]);

  const portfolioEcology = useMemo(() => {
    const usedCapacity = filteredPrograms.reduce((sum, p) => {
      return sum + (p.ecology?.used_capacity_MU || 0);
    }, 0);
    return {
      used_capacity_MU: usedCapacity,
      annual_share_percent: annualCapacity && annualCapacity > 0 
        ? (usedCapacity / annualCapacity) * 100 
        : null,
    };
  }, [filteredPrograms, annualCapacity]);

  if (loading) {
    return (
      <div className="card p-6">
        <h2 className="text-lg font-bold mb-4">Portfolio Dashboard</h2>
        <p className="text-gray-500 dark:text-gray-400">Loading portfolio data...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card p-6">
        <h2 className="text-lg font-bold mb-4">Portfolio Dashboard</h2>
        <p className="text-red-600 dark:text-red-400">{error}</p>
      </div>
    );
  }

  if (programData.length === 0) {
    return (
      <div className="card p-6">
        <h2 className="text-lg font-bold mb-4">Portfolio Dashboard</h2>
        <p className="text-gray-500 dark:text-gray-400">
          No program reports available. Sync programs to see aggregated work profile.
        </p>
      </div>
    );
  }

  const totalAlignment = aggregated.gyroscope.totals.GMT + aggregated.gyroscope.totals.ICV + 
                         aggregated.gyroscope.totals.IIA + aggregated.gyroscope.totals.ICI;
  const totalDisplacement = aggregated.thm.totals.GTD + aggregated.thm.totals.IVD + 
                            aggregated.thm.totals.IAD + aggregated.thm.totals.IID;
  const totalIncidents = totalAlignment + totalDisplacement;
  const alignmentRatio = totalIncidents > 0 ? (totalAlignment / totalIncidents) * 100 : 0;

  const exportJSON = () => {
    const exportData = filteredPrograms.map((p) => {
      const align = p.accounting.gyroscope.totals;
      const disp = p.accounting.thm.totals;
      const totalAlign = align.GMT + align.ICV + align.IIA + align.ICI;
      const totalDisp = disp.GTD + disp.IVD + disp.IAD + disp.IID;
      const total = totalAlign + totalDisp;
      const alignRatio = total > 0 ? (totalAlign / total) * 100 : 0;

      return {
        slug: p.slug,
        alignment: {
          total: totalAlign,
          ratio: alignRatio.toFixed(2),
          breakdown: align,
        },
        displacement: {
          total: totalDisp,
          breakdown: disp,
        },
        total_incidents: total,
        by_domain: {
          economy: {
            alignment: p.accounting.gyroscope.by_domain.economy,
            displacement: p.accounting.thm.by_domain.economy,
          },
          employment: {
            alignment: p.accounting.gyroscope.by_domain.employment,
            displacement: p.accounting.thm.by_domain.employment,
          },
          education: {
            alignment: p.accounting.gyroscope.by_domain.education,
            displacement: p.accounting.thm.by_domain.education,
          },
        },
      };
    });

    const summary = {
      export_date: new Date().toISOString(),
      total_programs: filteredPrograms.length,
      portfolio_summary: {
        total_incidents: totalIncidents,
        alignment_ratio: alignmentRatio.toFixed(2),
        total_alignment: totalAlignment,
        total_displacement: totalDisplacement,
      },
      programs: exportData,
    };

    const blob = new Blob([JSON.stringify(summary, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `portfolio-report-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const exportCSV = () => {
    const headers = [
      'Program Slug',
      'Unit',
      'Total Incidents',
      'Alignment Total',
      'Alignment Ratio %',
      'Displacement Total',
      'Estimated Value (£)',
      'GMT', 'ICV', 'IIA', 'ICI',
      'GTD', 'IVD', 'IAD', 'IID',
      'Economy Alignment', 'Economy Displacement',
      'Employment Alignment', 'Employment Displacement',
      'Education Alignment', 'Education Displacement',
    ];

    const rows = filteredPrograms.map((p) => {
      const align = p.accounting.gyroscope.totals;
      const disp = p.accounting.thm.totals;
      const totalAlign = align.GMT + align.ICV + align.IIA + align.ICI;
      const totalDisp = disp.GTD + disp.IVD + disp.IAD + disp.IID;
      const total = totalAlign + totalDisp;
      const alignRatio = total > 0 ? (totalAlign / total) * 100 : 0;
      const rate = p.unit === 'daily' ? 120 : 480;
      // Only alignment units are billable; displacement is risk coverage
      const estimatedValue = totalAlign * rate;

      const econAlign = p.accounting.gyroscope.by_domain.economy.GMT + 
                        p.accounting.gyroscope.by_domain.economy.ICV +
                        p.accounting.gyroscope.by_domain.economy.IIA +
                        p.accounting.gyroscope.by_domain.economy.ICI;
      const econDisp = p.accounting.thm.by_domain.economy.GTD +
                       p.accounting.thm.by_domain.economy.IVD +
                       p.accounting.thm.by_domain.economy.IAD +
                       p.accounting.thm.by_domain.economy.IID;
      const empAlign = p.accounting.gyroscope.by_domain.employment.GMT +
                       p.accounting.gyroscope.by_domain.employment.ICV +
                       p.accounting.gyroscope.by_domain.employment.IIA +
                       p.accounting.gyroscope.by_domain.employment.ICI;
      const empDisp = p.accounting.thm.by_domain.employment.GTD +
                      p.accounting.thm.by_domain.employment.IVD +
                      p.accounting.thm.by_domain.employment.IAD +
                      p.accounting.thm.by_domain.employment.IID;
      const eduAlign = p.accounting.gyroscope.by_domain.education.GMT +
                       p.accounting.gyroscope.by_domain.education.ICV +
                       p.accounting.gyroscope.by_domain.education.IIA +
                       p.accounting.gyroscope.by_domain.education.ICI;
      const eduDisp = p.accounting.thm.by_domain.education.GTD +
                      p.accounting.thm.by_domain.education.IVD +
                      p.accounting.thm.by_domain.education.IAD +
                      p.accounting.thm.by_domain.education.IID;

      return [
        p.slug,
        p.unit,
        total,
        totalAlign,
        alignRatio.toFixed(2),
        totalDisp,
        estimatedValue,
        align.GMT, align.ICV, align.IIA, align.ICI,
        disp.GTD, disp.IVD, disp.IAD, disp.IID,
        econAlign, econDisp,
        empAlign, empDisp,
        eduAlign, eduDisp,
      ];
    });

    const csvContent = [
      headers.join(','),
      ...rows.map((row) => row.join(',')),
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `portfolio-report-${new Date().toISOString().split('T')[0]}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      <div className="card p-6">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-lg font-bold">Portfolio Dashboard</h2>
            <div className="text-sm text-gray-500 dark:text-gray-400 mt-1">
              Estimated Value: <span className="font-mono font-bold text-gray-900 dark:text-white">£{totalValue.toLocaleString()}</span>
            </div>
            <div className="text-xs text-gray-400 dark:text-gray-500 mt-1">
              Alignment units are counted as work outputs for budgeting. Displacement units are risk coverage, not billable work.
            </div>
          </div>
          <div className="flex gap-2">
            <button
              type="button"
              className="btn btn-secondary btn-sm"
              onClick={exportJSON}
            >
              Export JSON
            </button>
            <button
              type="button"
              className="btn btn-secondary btn-sm"
              onClick={exportCSV}
            >
              Export CSV
            </button>
          </div>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-6">
          <div>
            <div className="text-xs text-gray-500 dark:text-gray-400">Active Programs</div>
            <div className="text-2xl font-bold">{filteredPrograms.length}</div>
            {filteredPrograms.length !== programData.length && (
              <div className="text-xs text-gray-400">of {programData.length}</div>
            )}
          </div>
          <div>
            <div className="text-xs text-gray-500 dark:text-gray-400">Total Incidents</div>
            <div className="text-2xl font-bold">{totalIncidents}</div>
          </div>
          <div>
            <div className="text-xs text-gray-500 dark:text-gray-400">Alignment Ratio</div>
            <div className="text-2xl font-bold text-emerald-600 dark:text-emerald-400">
              {alignmentRatio.toFixed(1)}%
            </div>
          </div>
          <div>
            <div className="text-xs text-gray-500 dark:text-gray-400">Displacement Units</div>
            <div className="text-2xl font-bold text-red-600 dark:text-red-400">{totalDisplacement}</div>
          </div>
        </div>

        {portfolioEcology.used_capacity_MU > 0 && (
          <div className="p-4 rounded-lg bg-indigo-50 dark:bg-indigo-950/30 border border-indigo-200 dark:border-indigo-800 mb-6">
            <div className="font-medium text-sm text-indigo-900 dark:text-indigo-100 mb-2">
              Portfolio Ecology Capacity
            </div>
            <div className="text-xs text-indigo-700 dark:text-indigo-300 space-y-1">
              <div>
                Total used capacity: <span className="font-mono font-semibold">{portfolioEcology.used_capacity_MU.toLocaleString()} MU</span>
              </div>
              {portfolioEcology.annual_share_percent !== null && (
                <div>
                  Portfolio currently uses{' '}
                  <span className="font-mono font-semibold">
                    {portfolioEcology.annual_share_percent.toExponential(2)}%
                  </span>{' '}
                  of annual capacity
                </div>
              )}
            </div>
          </div>
        )}

        <div className="pt-4 border-t border-gray-200 dark:border-gray-700">
          <div className="flex flex-col sm:flex-row gap-3">
            <input
              type="text"
              placeholder="Search programs..."
              className="input flex-1"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
            <select
              className="input"
              value={domainFilter}
              onChange={(e) => setDomainFilter(e.target.value as DomainKey | 'all')}
            >
              <option value="all">All Domains</option>
              <option value="economy">Economy</option>
              <option value="employment">Employment</option>
              <option value="education">Education</option>
            </select>
            <label className="flex items-center gap-2 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={highDisplacementFilter}
                onChange={(e) => setHighDisplacementFilter(e.target.checked)}
                className="rounded text-indigo-600"
              />
              <span className="text-sm">High Displacement Ratio</span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={multiDomainFilter}
                onChange={(e) => setMultiDomainFilter(e.target.checked)}
                className="rounded text-indigo-600"
              />
              <span className="text-sm">Multi-Domain Only</span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={onlyVerifiedFilter}
                onChange={(e) => setOnlyVerifiedFilter(e.target.checked)}
                className="rounded text-indigo-600"
              />
              <span className="text-sm">Only Verified</span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={onlySignedFilter}
                onChange={(e) => setOnlySignedFilter(e.target.checked)}
                className="rounded text-indigo-600"
              />
              <span className="text-sm">Only Signed</span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={onlyRealModeFilter}
                onChange={(e) => setOnlyRealModeFilter(e.target.checked)}
                className="rounded text-indigo-600"
              />
              <span className="text-sm">Real Mode Only</span>
            </label>
          </div>
          {filteredPrograms.length !== programData.length && (
            <div className="text-sm text-gray-500 dark:text-gray-400 mt-2">
              Showing {filteredPrograms.length} of {programData.length} programs
            </div>
          )}
        </div>
      </div>

      <WorkProfilePanel accounting={aggregated} />

      {filteredPrograms.length > 0 && (
        <div className="card overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-sm text-left">
              <thead className="bg-gray-50 dark:bg-gray-800 text-xs uppercase text-gray-500 dark:text-gray-400 font-semibold">
                <tr>
                  <th 
                    className="px-6 py-3 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700 select-none"
                    onClick={() => handleSort('slug')}
                  >
                    Program {sortColumn === 'slug' && (sortDirection === 'asc' ? '↑' : '↓')}
                  </th>
                  <th 
                    className="px-6 py-3 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700 select-none"
                    onClick={() => handleSort('unit')}
                  >
                    Unit {sortColumn === 'unit' && (sortDirection === 'asc' ? '↑' : '↓')}
                  </th>
                  <th 
                    className="px-6 py-3 text-right cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700 select-none"
                    onClick={() => handleSort('incidents')}
                  >
                    Incidents {sortColumn === 'incidents' && (sortDirection === 'asc' ? '↑' : '↓')}
                  </th>
                  <th 
                    className="px-6 py-3 text-right cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700 select-none"
                    onClick={() => handleSort('ratio')}
                  >
                    Ratio {sortColumn === 'ratio' && (sortDirection === 'asc' ? '↑' : '↓')}
                  </th>
                  <th 
                    className="px-6 py-3 text-right cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700 select-none"
                    onClick={() => handleSort('value')}
                  >
                    Est. Value {sortColumn === 'value' && (sortDirection === 'asc' ? '↑' : '↓')}
                  </th>
                  <th className="px-6 py-3 text-center">Signed</th>
                  <th className="px-6 py-3 text-center">Verified</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100 dark:divide-gray-800">
                {sortedPrograms.map((p) => {
                  const align = p.accounting.gyroscope.totals;
                  const disp = p.accounting.thm.totals;
                  const tAlign = align.GMT + align.ICV + align.IIA + align.ICI;
                  const tDisp = disp.GTD + disp.IVD + disp.IAD + disp.IID;
                  const total = tAlign + tDisp;
                  const ratio = total > 0 ? (tAlign / total) * 100 : 0;
                  const rate = p.unit === 'daily' ? 120 : 480;
                  // Only alignment units are billable; displacement is risk coverage
                  const val = tAlign * rate;

                  return (
                    <tr 
                      key={p.slug} 
                      onClick={() => onSelectProgram(p.slug)}
                      className="hover:bg-gray-50 dark:hover:bg-gray-800/50 cursor-pointer transition-colors"
                    >
                      <td className="px-6 py-3 font-medium text-indigo-600 dark:text-indigo-400">
                        {p.slug}
                      </td>
                      <td className="px-6 py-3 capitalize text-gray-500 dark:text-gray-400">
                        {p.unit}
                      </td>
                      <td className="px-6 py-3 text-right font-mono">
                        {total}
                      </td>
                      <td className="px-6 py-3 text-right">
                        <span className={ratio < 90 ? "text-amber-600 dark:text-amber-400" : "text-emerald-600 dark:text-emerald-400"}>
                          {ratio.toFixed(0)}%
                        </span>
                      </td>
                      <td className="px-6 py-3 text-right font-mono text-gray-600 dark:text-gray-300">
                        £{val.toLocaleString()}
                      </td>
                      <td className="px-6 py-3 text-center">
                        {p.signed ? (
                          <span className="badge bg-indigo-100 text-indigo-800 dark:bg-indigo-900 dark:text-indigo-200">Yes</span>
                        ) : (
                          <span className="text-gray-400">No</span>
                        )}
                      </td>
                      <td className="px-6 py-3 text-center">
                        {p.verified ? (
                          <span className="badge bg-emerald-100 text-emerald-800 dark:bg-emerald-900 dark:text-emerald-200">Yes</span>
                        ) : (
                          <span className="text-gray-400">No</span>
                        )}
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
  );
}

