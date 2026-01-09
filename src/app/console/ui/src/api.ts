import type {
  ProgramListResponse,
  ProgramResponse,
  EditableState,
  Glossary,
} from './types';

const BASE = '/api';

export async function listPrograms(): Promise<ProgramListResponse> {
  const res = await fetch(`${BASE}/programs`);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Failed to list programs: ${res.status} ${text}`);
  }
  return res.json();
}

export async function getProgram(slug: string): Promise<ProgramResponse> {
  const res = await fetch(`${BASE}/programs/${slug}`);
  if (!res.ok) throw new Error('Failed to load program');
  return res.json();
}

export async function createProgram(
  slug: string
): Promise<{ status: string; slug: string; program_id: string }> {
  const res = await fetch(`${BASE}/programs`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ slug }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'Failed to create program');
  }
  return res.json();
}

export async function updateProgram(
  slug: string,
  data: {
    unit: string;
    domain_counts: EditableState['domain_counts'];
    principle_counts: EditableState['principle_counts'];
    notes: string;
    agents: string;
    agencies: string;
  }
): Promise<ProgramResponse> {
  const res = await fetch(`${BASE}/programs/${slug}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error('Failed to update program');
  return res.json();
}

export async function deleteProgram(slug: string): Promise<{ status: string }> {
  const res = await fetch(`${BASE}/programs/${slug}`, { method: 'DELETE' });
  if (!res.ok) throw new Error('Failed to delete program');
  return res.json();
}

export function bundleUrl(slug: string): string {
  return `${BASE}/programs/${slug}/bundle`;
}

export async function syncProgram(slug: string): Promise<ProgramResponse> {
  const res = await fetch(`${BASE}/programs/${slug}/sync`, { method: 'POST' });
  if (!res.ok) throw new Error('Failed to sync program');
  return res.json();
}

export async function getGlossary(): Promise<Glossary> {
  const res = await fetch(`${BASE}/glossary`);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Failed to load glossary: ${res.status} ${text}`);
  }
  return res.json();
}

