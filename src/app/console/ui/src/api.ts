import type {
  ProjectListResponse,
  ProjectResponse,
  EditableState,
  Glossary,
} from './types';

const BASE = '/api';

export async function listProjects(): Promise<ProjectListResponse> {
  const res = await fetch(`${BASE}/projects`);
  if (!res.ok) throw new Error('Failed to list projects');
  return res.json();
}

export async function getProject(slug: string): Promise<ProjectResponse> {
  const res = await fetch(`${BASE}/projects/${slug}`);
  if (!res.ok) throw new Error('Failed to load project');
  return res.json();
}

export async function createProject(
  slug: string
): Promise<{ status: string; slug: string; project_id: string }> {
  const res = await fetch(`${BASE}/projects`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ slug }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'Failed to create project');
  }
  return res.json();
}

export async function updateProject(
  slug: string,
  data: {
    unit: string;
    domain_counts: EditableState['domain_counts'];
    principle_counts: EditableState['principle_counts'];
    notes: string;
  }
): Promise<ProjectResponse> {
  const res = await fetch(`${BASE}/projects/${slug}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error('Failed to update project');
  return res.json();
}

export async function deleteProject(slug: string): Promise<{ status: string }> {
  const res = await fetch(`${BASE}/projects/${slug}`, { method: 'DELETE' });
  if (!res.ok) throw new Error('Failed to delete project');
  return res.json();
}

export function bundleUrl(slug: string): string {
  return `${BASE}/projects/${slug}/bundle`;
}

export async function getGlossary(): Promise<Glossary> {
  const res = await fetch(`${BASE}/glossary`);
  if (!res.ok) throw new Error('Failed to load glossary');
  return res.json();
}

