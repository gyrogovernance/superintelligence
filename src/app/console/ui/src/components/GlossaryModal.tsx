import { useEffect, useRef } from 'react';
import type { Glossary } from '../types';

interface GlossaryModalProps {
  isOpen: boolean;
  onClose: () => void;
  glossary: Glossary | null;
  type: 'domain' | 'alignment' | 'displacement';
  itemKey: string;
}

export function GlossaryModal({
  isOpen,
  onClose,
  glossary,
  type,
  itemKey,
}: GlossaryModalProps) {
  const dialogRef = useRef<HTMLDivElement>(null);
  const closeBtnRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (isOpen) {
      closeBtnRef.current?.focus();
    }
  }, [isOpen]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen || !glossary) return null;

  const handleBackdropClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  let entry;
  if (type === 'domain') {
    entry = glossary.domains[itemKey];
  } else if (type === 'alignment') {
    entry = glossary.alignment[itemKey];
  } else {
    entry = glossary.displacement[itemKey];
  }

  if (!entry) return null;

  const getCategoryLabel = () => {
    switch (type) {
      case 'domain':
        return 'Domain';
      case 'alignment':
        return 'Alignment Principle';
      case 'displacement':
        return 'Displacement Risk';
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4 sm:p-6"
      onClick={handleBackdropClick}
      role="dialog"
      aria-modal="true"
      aria-labelledby="glossary-title"
    >
      <div
        ref={dialogRef}
        className="card w-full max-w-lg p-6 space-y-4 max-h-[90vh] overflow-y-auto"
      >
        <div className="flex items-start justify-between gap-4">
          <div>
            <span className="text-sm font-medium text-gray-500 dark:text-gray-400">
              {getCategoryLabel()}
            </span>
            <h2 id="glossary-title" className="text-xl font-bold">
              {type !== 'domain' && (
                <span className="text-indigo-600 dark:text-indigo-400">{itemKey}</span>
              )}{' '}
              {entry.name}
            </h2>
          </div>
          <button
            ref={closeBtnRef}
            type="button"
            className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
            onClick={onClose}
            aria-label="Close"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <p className="text-gray-600 dark:text-gray-300">{entry.description}</p>

        {entry.risk && (
          <div className="p-3 rounded-lg bg-amber-50 dark:bg-amber-950/50 border border-amber-200 dark:border-amber-800">
            <span className="text-sm font-medium text-amber-800 dark:text-amber-200">
              Risk: {entry.risk}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

