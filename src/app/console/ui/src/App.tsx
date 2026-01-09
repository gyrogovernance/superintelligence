import { useState, useEffect, useCallback, useRef } from 'react';
import type {
  Theme,
  AppStatus,
  ProgramSummary,
  ProgramResponse,
  EditableState,
  Glossary,
  DomainKey,
  PrincipleKey,
} from './types';
import { initTheme, applyTheme, getStoredTheme } from './theme';
import * as api from './api';

import { Header } from './components/Header';
import { ProgramBar } from './components/ProgramBar';
import { WorkProfilePanel } from './components/WorkProfilePanel';
import { ProgramSummaryCard } from './components/ProgramSummary';
import { ReportPanel } from './components/ReportPanel';
import { PortfolioView } from './components/PortfolioView';
import { SettingsPanel } from './components/SettingsPanel';
import { GlossaryModal } from './components/GlossaryModal';
import { ConfirmModal } from './components/ConfirmModal';

export default function App() {
  // State
  const [programs, setPrograms] = useState<ProgramSummary[]>([]);
  const [selectedSlug, setSelectedSlug] = useState<string | null>(null);
  const [program, setProgram] = useState<ProgramResponse | null>(null);
  const [glossary, setGlossary] = useState<Glossary | null>(null);
  const [status, setStatus] = useState<AppStatus>('idle');
  const [error, setError] = useState<string | null>(null);
  const [theme, setTheme] = useState<Theme>(getStoredTheme);
  const [viewMode, setViewMode] = useState<'program' | 'portfolio' | 'settings'>('program');

  // Modal state
  const [glossaryModal, setGlossaryModal] = useState<{
    isOpen: boolean;
    type: 'domain' | 'alignment' | 'displacement';
    itemKey: string;
  }>({ isOpen: false, type: 'domain', itemKey: '' });
  const [deleteConfirm, setDeleteConfirm] = useState(false);

  // Debounce timer ref
  const saveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Editable state (local copy for immediate UI updates)
  const [editable, setEditable] = useState<EditableState | null>(null);
  const [notes, setNotes] = useState<string>('');

  // Initialize theme
  useEffect(() => {
    initTheme();
  }, []);

  // Load programs and glossary on mount
  useEffect(() => {
    async function load() {
      setStatus('loading');
      try {
        const [programList, glossaryData] = await Promise.all([
          api.listPrograms(),
          api.getGlossary(),
        ]);
        setPrograms(programList.programs);
        setGlossary(glossaryData);
        setStatus('idle');
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load');
        setStatus('error');
      }
    }
    load();
  }, []);

  // Load program when selected
  useEffect(() => {
    if (!selectedSlug) {
      setProgram(null);
      setEditable(null);
      setNotes('');
      return;
    }

    async function loadProgram() {
      setStatus('loading');
      try {
        const data = await api.getProgram(selectedSlug!);
        setProgram(data);
        // Ensure agents and agencies are strings, not undefined
        // Clean up placeholder text if it's in the values
        const agentsValue = data.editable.agents || '';
        const agenciesValue = data.editable.agencies || '';
        const cleanedAgents = agentsValue === "(Names of people involved in this program)" ? '' : agentsValue;
        const cleanedAgencies = agenciesValue === "(Names of agencies involved in this program)" ? '' : agenciesValue;
        setEditable({
          ...data.editable,
          agents: cleanedAgents,
          agencies: cleanedAgencies,
        });
        setNotes(data.editable.notes || '');
        setStatus('idle');
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load program');
        setStatus('error');
      }
    }
    loadProgram();
  }, [selectedSlug]);

  // Theme handler
  const handleThemeChange = useCallback((newTheme: Theme) => {
    setTheme(newTheme);
    applyTheme(newTheme);
  }, []);

  // Save program - no debounce, called directly
  const saveProgram = useCallback(async (data: Partial<EditableState> & { notes?: string }) => {
    if (!selectedSlug || !editable) return;

    const payload = {
      unit: data.unit ?? editable.unit,
      domain_counts: data.domain_counts ?? editable.domain_counts,
      principle_counts: data.principle_counts ?? editable.principle_counts,
      notes: data.notes ?? notes,
      agents: data.agents ?? editable.agents ?? '',
      agencies: data.agencies ?? editable.agencies ?? '',
    };

    setStatus('saving');
    try {
      const updated = await api.updateProgram(selectedSlug, payload);
      setProgram(updated);
      setStatus('idle');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save');
      setStatus('error');
    }
  }, [selectedSlug, editable, notes]);

  // Debounced save for steppers
  const debouncedSave = useCallback((newEditable: EditableState) => {
    if (saveTimerRef.current) {
      clearTimeout(saveTimerRef.current);
    }
    saveTimerRef.current = setTimeout(() => {
      saveProgram(newEditable);
    }, 500);
  }, [saveProgram]);

  // Handlers
  const handleCreateProgram = useCallback(async (slug: string) => {
    setStatus('loading');
    try {
      const result = await api.createProgram(slug);
      const programList = await api.listPrograms();
      setPrograms(programList.programs);
      setSelectedSlug(result.slug);
      setViewMode('program');
      setStatus('idle');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create program');
      setStatus('error');
    }
  }, []);

  const handleDeleteProgram = useCallback(async () => {
    if (!selectedSlug) return;

    setStatus('loading');
    setDeleteConfirm(false);
    try {
      await api.deleteProgram(selectedSlug);
      const programList = await api.listPrograms();
      setPrograms(programList.programs);
      setSelectedSlug(null);
      setProgram(null);
      setEditable(null);
      setStatus('idle');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete program');
      setStatus('error');
    }
  }, [selectedSlug]);

  const handleUnitChange = useCallback((unit: 'daily' | 'sprint') => {
    if (!editable) return;
    const newEditable = { ...editable, unit };
    setEditable(newEditable);
    saveProgram({ unit });
  }, [editable, saveProgram]);

  const handleDomainChange = useCallback((domain: DomainKey, value: number) => {
    if (!editable) return;
    const newEditable = {
      ...editable,
      domain_counts: { ...editable.domain_counts, [domain]: value },
    };
    setEditable(newEditable);
    debouncedSave(newEditable);
  }, [editable, debouncedSave]);

  const handlePrincipleChange = useCallback((principle: PrincipleKey, value: number) => {
    if (!editable) return;
    const newEditable = {
      ...editable,
      principle_counts: { ...editable.principle_counts, [principle]: value },
    };
    setEditable(newEditable);
    debouncedSave(newEditable);
  }, [editable, debouncedSave]);

  const handleDomainInfo = useCallback((domain: string) => {
    setGlossaryModal({ isOpen: true, type: 'domain', itemKey: domain });
  }, []);

  const handlePrincipleInfo = useCallback((principle: string) => {
    const isAlignment = ['GMT', 'ICV', 'IIA', 'ICI'].includes(principle);
    setGlossaryModal({
      isOpen: true,
      type: isAlignment ? 'alignment' : 'displacement',
      itemKey: principle,
    });
  }, []);

  // Text field handlers - save immediately on blur (no debounce needed, local state in ReportPanel handles typing)
  const handleNotesChange = useCallback((value: string) => {
    setNotes(value);
    if (editable) {
      saveProgram({ notes: value });
    }
  }, [editable, saveProgram]);

  const handleAgentsChange = useCallback((value: string) => {
    if (!editable) return;
    const newEditable = { ...editable, agents: value };
    setEditable(newEditable);
    saveProgram({ agents: value });
  }, [editable, saveProgram]);

  const handleAgenciesChange = useCallback((value: string) => {
    if (!editable) return;
    const newEditable = { ...editable, agencies: value };
    setEditable(newEditable);
    saveProgram({ agencies: value });
  }, [editable, saveProgram]);

  const handleDownloadBundle = useCallback(() => {
    if (selectedSlug) {
      window.open(api.bundleUrl(selectedSlug), '_blank');
    }
  }, [selectedSlug]);

  // Cleanup timer on unmount
  useEffect(() => {
    return () => {
      if (saveTimerRef.current) {
        clearTimeout(saveTimerRef.current);
      }
    };
  }, []);

  const hasProgram = !!program && !!editable;

  return (
    <div className="min-h-screen">
      <Header
        theme={theme}
        onThemeChange={handleThemeChange}
        status={status}
        hasProgram={hasProgram}
        viewMode={viewMode}
        onViewModeChange={setViewMode}
      />

      <main className="max-w-7xl mx-auto px-3 py-3">
        {error && (
          <div className="mb-4 p-4 rounded-lg bg-red-50 dark:bg-red-950/50 border border-red-200 dark:border-red-800">
            <div className="flex items-center justify-between">
              <span className="text-red-700 dark:text-red-300">{error}</span>
              <button
                type="button"
                className="text-red-500 hover:text-red-700"
                onClick={() => setError(null)}
              >
                Dismiss
              </button>
            </div>
          </div>
        )}

        {/* Program Bar */}
        <ProgramBar
          programs={programs}
          selectedSlug={selectedSlug}
          onSelectProgram={setSelectedSlug}
          onCreateProgram={handleCreateProgram}
        />

        {viewMode === 'portfolio' ? (
          <PortfolioView 
            programs={programs} 
            onSelectProgram={(slug) => {
              setSelectedSlug(slug);
              setViewMode('program');
            }} 
          />
        ) : viewMode === 'settings' ? (
          <SettingsPanel
            program={program}
            selectedSlug={selectedSlug}
            onProgramReload={async () => {
              if (selectedSlug) {
                const data = await api.getProgram(selectedSlug);
                setProgram(data);
                // Clean up placeholder text if it's in the values
                const agentsValue = data.editable.agents || '';
                const agenciesValue = data.editable.agencies || '';
                const cleanedAgents = agentsValue === "(Names of people involved in this program)" ? '' : agentsValue;
                const cleanedAgencies = agenciesValue === "(Names of agencies involved in this program)" ? '' : agenciesValue;
                setEditable({
                  ...data.editable,
                  agents: cleanedAgents,
                  agencies: cleanedAgencies,
                });
                setNotes(data.editable.notes || '');
              }
            }}
            unit={editable?.unit || 'daily'}
            onUnitChange={handleUnitChange}
          />
        ) : !hasProgram ? (
          <div className="text-center py-16">
            <div className="w-20 h-20 mx-auto mb-6 rounded-2xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-lg">
              <span className="text-white text-4xl font-bold">🍃</span>
            </div>
            <h2 className="text-2xl font-bold mb-2">Welcome to AIR Console</h2>
            <p className="text-gray-500 dark:text-gray-400 mb-6 max-w-md mx-auto">
              Select an existing program or create a new one to begin managing
              AI governance contracts.
            </p>
            {programs.length === 0 && status === 'idle' && (
              <p className="text-sm text-gray-400 dark:text-gray-500">
                No programs found. Click "+ New" above to create your first program.
              </p>
            )}
          </div>
        ) : (
          <>
            {program.report && (
              <ProgramSummaryCard
                compilation={program.report.compilation}
                accounting={program.report.accounting}
                unit={editable?.unit || 'daily'}
                lastSynced={program.last_synced}
                domainCounts={editable?.domain_counts}
              />
            )}
            <div className="grid gap-6 lg:grid-cols-5">
              {/* Left Column - Work Profile */}
              <div className="lg:col-span-2">
                <WorkProfilePanel
                  accounting={program.report?.accounting || null}
                />
              </div>

              {/* Right Column - Report */}
              <div className="lg:col-span-3">
                <ReportPanel
                  report={program.report}
                  onDownloadBundle={handleDownloadBundle}
                  notes={notes}
                  onNotesChange={handleNotesChange}
                  editable={editable}
                  onDomainChange={handleDomainChange}
                  onPrincipleChange={handlePrincipleChange}
                  onDomainInfo={handleDomainInfo}
                  onPrincipleInfo={handlePrincipleInfo}
                  onAgentsChange={handleAgentsChange}
                  onAgenciesChange={handleAgenciesChange}
                  hasEventLog={program.has_event_log ?? false}
                />
              </div>
            </div>
          </>
        )}
      </main>

      {/* Modals */}
      <GlossaryModal
        isOpen={glossaryModal.isOpen}
        onClose={() => setGlossaryModal({ ...glossaryModal, isOpen: false })}
        glossary={glossary}
        type={glossaryModal.type}
        itemKey={glossaryModal.itemKey}
      />

      <ConfirmModal
        isOpen={deleteConfirm}
        title="Delete Program"
        message={`Are you sure you want to delete "${selectedSlug}"? This will remove the program file and all compiled artifacts. This action cannot be undone.`}
        confirmLabel="Delete"
        variant="danger"
        onConfirm={handleDeleteProgram}
        onCancel={() => setDeleteConfirm(false)}
      />
    </div>
  );
}

