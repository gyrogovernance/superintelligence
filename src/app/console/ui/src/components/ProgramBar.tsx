import { useState, useRef, useEffect } from 'react';
import type { ProgramSummary } from '../types';

interface ProgramBarProps {
  programs: ProgramSummary[];
  selectedSlug: string | null;
  onSelectProgram: (slug: string | null) => void;
  onCreateProgram: (slug: string) => void;
}

export function ProgramBar({
  programs,
  selectedSlug,
  onSelectProgram,
  onCreateProgram,
}: ProgramBarProps) {
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newSlug, setNewSlug] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (showCreateModal && inputRef.current) {
      inputRef.current.focus();
    }
  }, [showCreateModal]);

  const handleCreate = () => {
    if (newSlug.trim()) {
      onCreateProgram(newSlug.trim().toLowerCase().replace(/[^a-z0-9-]/g, '-'));
      setNewSlug('');
      setShowCreateModal(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleCreate();
    } else if (e.key === 'Escape') {
      setShowCreateModal(false);
      setNewSlug('');
    }
  };

  return (
    <div className="flex flex-wrap items-center justify-between gap-3 mb-6 p-4 rounded-xl bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 shadow-sm">
      {/* Left side: Program Selector and New Button */}
      <div className="flex flex-wrap items-center gap-2">
        <label htmlFor="program-select" className="text-sm font-medium text-gray-600 dark:text-gray-400">
          Program:
        </label>
        <select
          id="program-select"
          className="select min-w-[160px]"
          value={selectedSlug || ''}
          onChange={(e) => onSelectProgram(e.target.value || null)}
          aria-label="Select program"
        >
          <option value="">Select program...</option>
          {programs.map((p) => (
            <option key={p.slug} value={p.slug}>
              {p.slug}
            </option>
          ))}
        </select>
        <button
          type="button"
          className="btn btn-primary btn-sm"
          onClick={() => setShowCreateModal(true)}
        >
          + New
        </button>
      </div>

      {/* Create Modal */}
      {showCreateModal && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
          onClick={(e) => {
            if (e.target === e.currentTarget) {
              setShowCreateModal(false);
              setNewSlug('');
            }
          }}
        >
          <div className="card w-full max-w-md p-6 space-y-4">
            <h2 className="text-xl font-bold">Create New Program</h2>
            <div className="space-y-2">
              <label htmlFor="new-slug" className="block text-sm font-medium">
                Program Slug
              </label>
              <input
                ref={inputRef}
                id="new-slug"
                type="text"
                className="input w-full"
                placeholder="my-program-name"
                value={newSlug}
                onChange={(e) => setNewSlug(e.target.value)}
                onKeyDown={handleKeyDown}
              />
              <p className="text-xs text-gray-500">
                Use lowercase letters, numbers, and hyphens only.
              </p>
            </div>
            <div className="flex justify-end gap-3">
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => {
                  setShowCreateModal(false);
                  setNewSlug('');
                }}
              >
                Cancel
              </button>
              <button
                type="button"
                className="btn btn-primary"
                onClick={handleCreate}
                disabled={!newSlug.trim()}
              >
                Create Program
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
