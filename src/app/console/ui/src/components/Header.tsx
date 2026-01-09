import type { Theme, AppStatus } from '../types';

interface HeaderProps {
  theme: Theme;
  onThemeChange: (theme: Theme) => void;
  status: AppStatus;
  hasProgram: boolean;
  viewMode: 'program' | 'portfolio' | 'settings';
  onViewModeChange: (mode: 'program' | 'portfolio' | 'settings') => void;
}

export function Header({
  theme,
  onThemeChange,
  status,
  hasProgram,
  viewMode,
  onViewModeChange,
}: HeaderProps) {
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
        <div className="flex items-center justify-between">
          {/* Logo and Title */}
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-md">
              <span className="text-white font-bold text-lg">🍃</span>
            </div>
            <div>
              <h1 className="text-xl font-bold">AIR Console</h1>
              <p className="text-xs text-gray-500 dark:text-gray-400">Governance Compiler</p>
            </div>
            
            {/* Status - moved to left side */}
            <div className="flex items-center gap-1.5 ml-2" aria-live="polite">
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
              {status === 'idle' && hasProgram && (
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

          {/* Right side: View Mode Dropdown + Theme */}
          <div className="flex items-center gap-3">
            {/* View Mode Dropdown */}
            <select
              className="select min-w-[180px]"
              value={viewMode}
              onChange={(e) => onViewModeChange(e.target.value as 'program' | 'portfolio' | 'settings')}
              aria-label="Select view mode"
            >
              <option value="program">Program View</option>
              <option value="portfolio">Portfolio Dashboard</option>
              {hasProgram && <option value="settings">Settings</option>}
            </select>

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
          </div>
        </div>
      </div>
    </header>
  );
}
