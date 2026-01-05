import { useState, useEffect, useMemo } from 'react';
import type { ProjectSummary, ReportAccounting, DomainKey } from '../types';
import * as api from '../api';
import { WorkProfilePanel } from './WorkProfilePanel';

interface PortfolioViewProps {
  projects: ProjectSummary[];
  onSelectProject: (slug: string) => void;
}

interface ProjectData {
  slug: string;
  accounting: ReportAccounting;
  unit: 'daily' | 'sprint';
  lastSynced: string | null;
}

function aggregateAccounting(data: ProjectData[]): ReportAccounting {
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

export function PortfolioView({ projects, onSelectProject }: PortfolioViewProps) {
  const [projectData, setProjectData] = useState<ProjectData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [domainFilter, setDomainFilter] = useState<DomainKey | 'all'>('all');
  const [highDisplacementFilter, setHighDisplacementFilter] = useState(false);
  const [sortColumn, setSortColumn] = useState<'slug' | 'unit' | 'incidents' | 'ratio' | 'value'>('slug');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc');

  useEffect(() => {
    async function loadAllReports() {
      setLoading(true);
      setError(null);
      try {
        const reports = await Promise.all(
          projects.map(async (p) => {
            try {
              const project = await api.getProject(p.slug);
              if (project.report?.accounting && project.editable) {
                return { 
                  slug: p.slug, 
                  accounting: project.report.accounting,
                  unit: project.editable.unit,
                  lastSynced: project.last_synced,
                };
              }
              return null;
            } catch {
              return null;
            }
          })
        );
        const validProjects = reports.filter((p): p is ProjectData => p !== null);
        setProjectData(validProjects);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load portfolio data');
      } finally {
        setLoading(false);
      }
    }

    if (projects.length > 0) {
      loadAllReports();
    } else {
      setProjectData([]);
      setLoading(false);
    }
  }, [projects]);

  const filteredProjects = useMemo(() => {
    let filtered = projectData;

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

    return filtered;
  }, [projectData, searchQuery, domainFilter, highDisplacementFilter]);

  const sortedProjects = useMemo(() => {
    const sorted = [...filteredProjects].sort((a, b) => {
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
        const valA = totalA * rateA;
        const ratioA = totalA > 0 ? (totalAlignA / totalA) * 100 : 0;

        const alignB = b.accounting.gyroscope.totals;
        const dispB = b.accounting.thm.totals;
        const totalAlignB = alignB.GMT + alignB.ICV + alignB.IIA + alignB.ICI;
        const totalDispB = dispB.GTD + dispB.IVD + dispB.IAD + dispB.IID;
        const totalB = totalAlignB + totalDispB;
        const rateB = b.unit === 'daily' ? 120 : 480;
        const valB = totalB * rateB;
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
  }, [filteredProjects, sortColumn, sortDirection]);

  const handleSort = (column: typeof sortColumn) => {
    if (sortColumn === column) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortColumn(column);
      setSortDirection('asc');
    }
  };

  const aggregated = useMemo(() => {
    return aggregateAccounting(filteredProjects);
  }, [filteredProjects]);

  const totalValue = useMemo(() => {
    return filteredProjects.reduce((sum, p) => {
      const align = p.accounting.gyroscope.totals;
      const disp = p.accounting.thm.totals;
      const totalIncidents = 
        disp.GTD + disp.IVD + disp.IAD + disp.IID +
        align.GMT + align.ICV + align.IIA + align.ICI;
      
      const rate = p.unit === 'daily' ? 120 : 480;
      return sum + (totalIncidents * rate);
    }, 0);
  }, [filteredProjects]);

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

  if (projectData.length === 0) {
    return (
      <div className="card p-6">
        <h2 className="text-lg font-bold mb-4">Portfolio Dashboard</h2>
        <p className="text-gray-500 dark:text-gray-400">
          No project reports available. Sync projects to see aggregated work profile.
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
    const exportData = filteredProjects.map((p) => {
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
      total_projects: filteredProjects.length,
      portfolio_summary: {
        total_incidents: totalIncidents,
        alignment_ratio: alignmentRatio.toFixed(2),
        total_alignment: totalAlignment,
        total_displacement: totalDisplacement,
      },
      projects: exportData,
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
      'Project Slug',
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

    const rows = filteredProjects.map((p) => {
      const align = p.accounting.gyroscope.totals;
      const disp = p.accounting.thm.totals;
      const totalAlign = align.GMT + align.ICV + align.IIA + align.ICI;
      const totalDisp = disp.GTD + disp.IVD + disp.IAD + disp.IID;
      const total = totalAlign + totalDisp;
      const alignRatio = total > 0 ? (totalAlign / total) * 100 : 0;
      const rate = p.unit === 'daily' ? 120 : 480;
      const estimatedValue = total * rate;

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
            <div className="text-xs text-gray-500 dark:text-gray-400">Active Projects</div>
            <div className="text-2xl font-bold">{filteredProjects.length}</div>
            {filteredProjects.length !== projectData.length && (
              <div className="text-xs text-gray-400">of {projectData.length}</div>
            )}
          </div>
          <div>
            <div className="text-xs text-gray-500 dark:text-gray-400">Total Units</div>
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

        <div className="pt-4 border-t border-gray-200 dark:border-gray-700">
          <div className="flex flex-col sm:flex-row gap-3">
            <input
              type="text"
              placeholder="Search projects..."
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
              <span className="text-sm">High Risk</span>
            </label>
          </div>
          {filteredProjects.length !== projectData.length && (
            <div className="text-sm text-gray-500 dark:text-gray-400 mt-2">
              Showing {filteredProjects.length} of {projectData.length} projects
            </div>
          )}
        </div>
      </div>

      <WorkProfilePanel accounting={aggregated} />

      {filteredProjects.length > 0 && (
        <div className="card overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-sm text-left">
              <thead className="bg-gray-50 dark:bg-gray-800 text-xs uppercase text-gray-500 dark:text-gray-400 font-semibold">
                <tr>
                  <th 
                    className="px-6 py-3 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700 select-none"
                    onClick={() => handleSort('slug')}
                  >
                    Project {sortColumn === 'slug' && (sortDirection === 'asc' ? '↑' : '↓')}
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
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100 dark:divide-gray-800">
                {sortedProjects.map((p) => {
                  const align = p.accounting.gyroscope.totals;
                  const disp = p.accounting.thm.totals;
                  const tAlign = align.GMT + align.ICV + align.IIA + align.ICI;
                  const tDisp = disp.GTD + disp.IVD + disp.IAD + disp.IID;
                  const total = tAlign + tDisp;
                  const ratio = total > 0 ? (tAlign / total) * 100 : 0;
                  const rate = p.unit === 'daily' ? 120 : 480;
                  const val = total * rate;

                  return (
                    <tr 
                      key={p.slug} 
                      onClick={() => onSelectProject(p.slug)}
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

