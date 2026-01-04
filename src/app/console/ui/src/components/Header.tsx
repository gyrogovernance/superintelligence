import { useState } from 'react';
import type { Theme, ProjectSummary, AppStatus } from '../types';

interface HeaderProps {
  theme: Theme;
  onThemeChange: (theme: Theme) => void;
  projects: ProjectSummary[];
  selectedSlug: string | null;
  onSelectProject: (slug: string | null) => void;
  onCreateProject: (slug: string) => void;
  onDeleteProject: () => void;
  unit: 'daily' | 'sprint';
  onUnitChange: (unit: 'daily' | 'sprint') => void;
  status: AppStatus;
  hasProject: boolean;
}

export function Header({
  theme,
  onThemeChange,
  projects,
  selectedSlug,
  onSelectProject,
  onCreateProject,
  onDeleteProject,
  unit,
  onUnitChange,
  status,
  hasProject,
}: HeaderProps) {
  const [showCreateInput, setShowCreateInput] = useState(false);
  const [newSlug, setNewSlug] = useState('');

  const handleCreate = () => {
    if (newSlug.trim()) {
      onCreateProject(newSlug.trim().toLowerCase());
      setNewSlug('');
      setShowCreateInput(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleCreate();
    } else if (e.key === 'Escape') {
      setShowCreateInput(false);
      setNewSlug('');
    }
  };

  const themeIcons = {
    light: (
      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
      </svg>
    ),
    dark: (
      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
      </svg>
    ),
    system: (
      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
      </svg>
    ),
  };

  const cycleTheme = () => {
    const themes: Theme[] = ['light', 'dark', 'system'];
    const currentIndex = themes.indexOf(theme);
    const nextTheme = themes[(currentIndex + 1) % themes.length];
    onThemeChange(nextTheme);
  };

  return (
    <header className="sticky top-0 z-40 bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm border-b border-gray-200 dark:border-gray-700">
      <div className="max-w-7xl mx-auto px-4 py-3">
        <div className="flex flex-col sm:flex-row sm:items-center gap-3">
          {/* Logo and Title */}
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-md">
              <span className="text-white font-bold text-lg">🍃</span>
            </div>
            <div>
              <h1 className="text-xl font-bold">AIR Console</h1>
              <p className="text-xs text-gray-500 dark:text-gray-400">Governance Compiler</p>
            </div>
          </div>

          {/* Controls */}
          <div className="flex flex-wrap items-center gap-2 sm:ml-auto">
            {/* Project Selector */}
            <select
              className="select min-w-[140px]"
              value={selectedSlug || ''}
              onChange={(e) => onSelectProject(e.target.value || null)}
              aria-label="Select project"
            >
              <option value="">Select project...</option>
              {projects.map((p) => (
                <option key={p.slug} value={p.slug}>
                  {p.slug}
                </option>
              ))}
            </select>

            {/* Create Project */}
            {showCreateInput ? (
              <div className="flex items-center gap-1">
                <input
                  type="text"
                  className="input w-32"
                  placeholder="project-slug"
                  value={newSlug}
                  onChange={(e) => setNewSlug(e.target.value)}
                  onKeyDown={handleKeyDown}
                  autoFocus
                />
                <button
                  type="button"
                  className="btn btn-primary btn-sm"
                  onClick={handleCreate}
                  disabled={!newSlug.trim()}
                >
                  Create
                </button>
                <button
                  type="button"
                  className="btn btn-secondary btn-sm"
                  onClick={() => {
                    setShowCreateInput(false);
                    setNewSlug('');
                  }}
                >
                  Cancel
                </button>
              </div>
            ) : (
              <button
                type="button"
                className="btn btn-primary btn-sm"
                onClick={() => setShowCreateInput(true)}
              >
                + New
              </button>
            )}

            {/* Delete Project */}
            {hasProject && (
              <button
                type="button"
                className="btn btn-danger btn-sm"
                onClick={onDeleteProject}
                aria-label="Delete project"
              >
                Delete
              </button>
            )}

            {/* Unit Toggle */}
            {hasProject && (
              <select
                className="select"
                value={unit}
                onChange={(e) => onUnitChange(e.target.value as 'daily' | 'sprint')}
                aria-label="Select time unit"
              >
                <option value="daily">Daily</option>
                <option value="sprint">Sprint</option>
              </select>
            )}

            {/* Theme Toggle */}
            <button
              type="button"
              className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
              onClick={cycleTheme}
              aria-label={`Theme: ${theme}`}
              title={`Theme: ${theme}`}
            >
              {themeIcons[theme]}
            </button>

            {/* Status */}
            <div className="flex items-center gap-1.5" aria-live="polite">
              {status === 'loading' && (
                <>
                  <div className="w-2 h-2 rounded-full bg-amber-500 animate-pulse" />
                  <span className="text-xs text-gray-500">Loading...</span>
                </>
              )}
              {status === 'saving' && (
                <>
                  <div className="w-2 h-2 rounded-full bg-indigo-500 animate-pulse" />
                  <span className="text-xs text-gray-500">Saving...</span>
                </>
              )}
              {status === 'idle' && hasProject && (
                <>
                  <div className="w-2 h-2 rounded-full bg-emerald-500" />
                  <span className="text-xs text-gray-500">Synced</span>
                </>
              )}
              {status === 'error' && (
                <>
                  <div className="w-2 h-2 rounded-full bg-red-500" />
                  <span className="text-xs text-red-500">Error</span>
                </>
              )}
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}

