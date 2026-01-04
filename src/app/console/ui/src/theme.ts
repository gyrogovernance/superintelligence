import type { Theme } from './types';

export function getStoredTheme(): Theme {
  const stored = localStorage.getItem('theme');
  if (stored === 'light' || stored === 'dark' || stored === 'system') {
    return stored;
  }
  return 'system';
}

export function applyTheme(theme: Theme): void {
  const root = document.documentElement;
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;

  root.classList.remove('light', 'dark');

  if (theme === 'dark' || (theme === 'system' && prefersDark)) {
    root.classList.add('dark');
  } else {
    root.classList.add('light');
  }

  localStorage.setItem('theme', theme);
}

export function initTheme(): void {
  applyTheme(getStoredTheme());

  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
    if (getStoredTheme() === 'system') {
      applyTheme('system');
    }
  });
}

