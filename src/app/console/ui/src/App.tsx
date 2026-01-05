import { useState, useEffect, useCallback, useRef } from 'react';
import type {
  Theme,
  AppStatus,
  ProjectSummary,
  ProjectResponse,
  EditableState,
  Glossary,
  DomainKey,
  PrincipleKey,
} from './types';
import { initTheme, applyTheme, getStoredTheme } from './theme';
import * as api from './api';

import { Header } from './components/Header';
import { WorkProfilePanel } from './components/WorkProfilePanel';
import { ProjectSummaryCard } from './components/ProjectSummary';
import { ReportPanel } from './components/ReportPanel';
import { PortfolioView } from './components/PortfolioView';
import { GlossaryModal } from './components/GlossaryModal';
import { ConfirmModal } from './components/ConfirmModal';

export default function App() {
  // State
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [selectedSlug, setSelectedSlug] = useState<string | null>(null);
  const [project, setProject] = useState<ProjectResponse | null>(null);
  const [glossary, setGlossary] = useState<Glossary | null>(null);
  const [status, setStatus] = useState<AppStatus>('idle');
  const [error, setError] = useState<string | null>(null);
  const [theme, setTheme] = useState<Theme>(getStoredTheme);
  const [viewMode, setViewMode] = useState<'project' | 'portfolio'>('project');

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

  // Load projects and glossary on mount
  useEffect(() => {
    async function load() {
      setStatus('loading');
      try {
        const [projectList, glossaryData] = await Promise.all([
          api.listProjects(),
          api.getGlossary(),
        ]);
        setProjects(projectList.projects);
        setGlossary(glossaryData);
        setStatus('idle');
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load');
        setStatus('error');
      }
    }
    load();
  }, []);

  // Load project when selected
  useEffect(() => {
    if (!selectedSlug) {
      setProject(null);
      setEditable(null);
      setNotes('');
      return;
    }

    async function loadProject() {
      setStatus('loading');
      try {
        const data = await api.getProject(selectedSlug!);
        setProject(data);
        setEditable(data.editable);
        setNotes(data.editable.notes || '');
        setStatus('idle');
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load project');
        setStatus('error');
      }
    }
    loadProject();
  }, [selectedSlug]);

  // Theme handler
  const handleThemeChange = useCallback((newTheme: Theme) => {
    setTheme(newTheme);
    applyTheme(newTheme);
  }, []);

  // Save project (debounced)
  const saveProject = useCallback(async (data: EditableState & { notes?: string }) => {
    if (!selectedSlug) return;

    setStatus('saving');
    try {
      const updated = await api.updateProject(selectedSlug, {
        unit: data.unit,
        domain_counts: data.domain_counts,
        principle_counts: data.principle_counts,
        notes: data.notes ?? notes,
      });
      setProject(updated);
      setEditable(updated.editable); // Keep editable in sync with backend
      setNotes(updated.editable.notes || '');
      setStatus('idle');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save');
      setStatus('error');
    }
  }, [selectedSlug, notes]);

  // Debounced save trigger
  const triggerSave = useCallback((newEditable: EditableState) => {
    if (saveTimerRef.current) {
      clearTimeout(saveTimerRef.current);
    }
    saveTimerRef.current = setTimeout(() => {
      saveProject(newEditable);
    }, 500);
  }, [saveProject]);

  // Handlers
  const handleCreateProject = useCallback(async (slug: string) => {
    setStatus('loading');
    try {
      const result = await api.createProject(slug);
      const projectList = await api.listProjects();
      setProjects(projectList.projects);
      setSelectedSlug(result.slug);
      setStatus('idle');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create project');
      setStatus('error');
    }
  }, []);

  const handleDeleteProject = useCallback(async () => {
    if (!selectedSlug) return;

    setStatus('loading');
    setDeleteConfirm(false);
    try {
      await api.deleteProject(selectedSlug);
      const projectList = await api.listProjects();
      setProjects(projectList.projects);
      setSelectedSlug(null);
      setProject(null);
      setEditable(null);
      setStatus('idle');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete project');
      setStatus('error');
    }
  }, [selectedSlug]);

  const handleUnitChange = useCallback((unit: 'daily' | 'sprint') => {
    if (!editable) return;
    const newEditable = { ...editable, unit };
    setEditable(newEditable);
    triggerSave(newEditable);
  }, [editable, triggerSave]);

  const handleDomainChange = useCallback((domain: DomainKey, value: number) => {
    if (!editable) return;
    const newEditable = {
      ...editable,
      domain_counts: { ...editable.domain_counts, [domain]: value },
    };
    setEditable(newEditable);
    triggerSave(newEditable);
  }, [editable, triggerSave]);

  const handlePrincipleChange = useCallback((principle: PrincipleKey, value: number) => {
    if (!editable) return;
    const newEditable = {
      ...editable,
      principle_counts: { ...editable.principle_counts, [principle]: value },
    };
    setEditable(newEditable);
    triggerSave(newEditable);
  }, [editable, triggerSave]);

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

  const handleNotesChange = useCallback((value: string) => {
    setNotes(value);
    if (!editable || !selectedSlug) return;
    triggerSave({ ...editable, notes: value });
  }, [editable, selectedSlug, triggerSave]);

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

  const hasProject = !!project && !!editable;

  return (
    <div className="min-h-screen">
      <Header
        theme={theme}
        onThemeChange={handleThemeChange}
        projects={projects}
        selectedSlug={selectedSlug}
        onSelectProject={setSelectedSlug}
        onCreateProject={handleCreateProject}
        onDeleteProject={() => setDeleteConfirm(true)}
        unit={editable?.unit || 'daily'}
        onUnitChange={handleUnitChange}
        status={status}
        hasProject={hasProject}
      />

      <main className="max-w-7xl mx-auto px-4 py-6">
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

        {/* View Mode Toggle */}
        <div className="mb-6 flex gap-2">
          <button
            type="button"
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              viewMode === 'project'
                ? 'bg-indigo-600 text-white'
                : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600'
            }`}
            onClick={() => setViewMode('project')}
          >
            Project View
          </button>
          <button
            type="button"
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              viewMode === 'portfolio'
                ? 'bg-indigo-600 text-white'
                : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600'
            }`}
            onClick={() => setViewMode('portfolio')}
          >
            Portfolio Dashboard
          </button>
        </div>

        {viewMode === 'portfolio' ? (
          <PortfolioView 
            projects={projects} 
            onSelectProject={(slug) => {
              setSelectedSlug(slug);
              setViewMode('project');
            }} 
          />
        ) : !hasProject ? (
          <div className="text-center py-16">
            <div className="w-20 h-20 mx-auto mb-6 rounded-2xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-lg">
              <span className="text-white text-4xl font-bold">🍃</span>
            </div>
            <h2 className="text-2xl font-bold mb-2">Welcome to AIR Console</h2>
            <p className="text-gray-500 dark:text-gray-400 mb-6 max-w-md mx-auto">
              Select an existing project or create a new one to begin managing
              AI governance contracts.
            </p>
            {projects.length === 0 && status === 'idle' && (
              <p className="text-sm text-gray-400 dark:text-gray-500">
                No projects found. Click "+ New" in the header to create your first project.
              </p>
            )}
          </div>
        ) : (
          <>
            {project.report && (
              <ProjectSummaryCard
                compilation={project.report.compilation}
                accounting={project.report.accounting}
                unit={editable?.unit || 'daily'}
                lastSynced={project.last_synced}
                onUnitChange={handleUnitChange}
              />
            )}
            <div className="grid gap-6 lg:grid-cols-5">
              {/* Left Column - Work Profile */}
              <div className="lg:col-span-2">
                <WorkProfilePanel
                  accounting={project.report?.accounting || null}
                />
              </div>

              {/* Right Column - Report */}
              <div className="lg:col-span-3">
                <ReportPanel
                  report={project.report}
                  onDownloadBundle={handleDownloadBundle}
                  notes={notes}
                  onNotesChange={handleNotesChange}
                  editable={editable}
                  onDomainChange={handleDomainChange}
                  onPrincipleChange={handlePrincipleChange}
                  onDomainInfo={handleDomainInfo}
                  onPrincipleInfo={handlePrincipleInfo}
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
        title="Delete Project"
        message={`Are you sure you want to delete "${selectedSlug}"? This will remove the project file and all compiled artifacts. This action cannot be undone.`}
        confirmLabel="Delete"
        variant="danger"
        onConfirm={handleDeleteProject}
        onCancel={() => setDeleteConfirm(false)}
      />
    </div>
  );
}

