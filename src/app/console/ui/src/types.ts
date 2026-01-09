// Matches store.parse_program_from_markdown output
export interface EditableState {
  slug: string;
  unit: 'daily' | 'sprint';
  domain_counts: {
    economy: number;
    employment: number;
    education: number;
  };
  principle_counts: {
    GMT: number;
    GTD: number;
    ICV: number;
    IVD: number;
    IIA: number;
    IAD: number;
    ICI: number;
    IID: number;
  };
  notes: string;
  agents: string;
  agencies: string;
}

// Type-safe keys for domains and principles
export type DomainKey = 'economy' | 'employment' | 'education';
export type PrincipleKey = keyof EditableState['principle_counts'];

// Matches .report.json structure exactly
export interface ReportKernel {
  step: number;
  state_index: number;
  state_hex: string;
  a_hex: string;
  b_hex: string;
  last_byte: number;
}

export interface ReportHashes {
  bytes_sha256: string;
  events_sha256: string;
}

export interface ReportCompilation {
  attestation_count: number;
  processed_attestations: number;
  skipped_attestations: Array<{ index: number; id: string | null; reason: string }>;
  byte_count: number;
  kernel: ReportKernel;
  hashes: ReportHashes;
}

export interface THMTotals {
  GTD: number;
  IVD: number;
  IAD: number;
  IID: number;
}

export interface GyroTotals {
  GMT: number;
  ICV: number;
  IIA: number;
  ICI: number;
}

export interface DomainBreakdown<T> {
  economy: T;
  employment: T;
  education: T;
}

export interface ReportAccounting {
  thm: {
    totals: THMTotals;
    by_domain: DomainBreakdown<THMTotals>;
  };
  gyroscope: {
    totals: GyroTotals;
    by_domain: DomainBreakdown<GyroTotals>;
  };
}

export interface ReportLedger {
  y_econ: number[];
  y_emp: number[];
  y_edu: number[];
}

export interface ReportApertures {
  A_econ: number;
  A_emp: number;
  A_edu: number;
}

export interface Report {
  program_slug: string;
  program_id: string;
  compilation: ReportCompilation;
  accounting: ReportAccounting;
  ledger: ReportLedger;
  apertures: ReportApertures;
  warnings: Record<string, unknown>;
}

// API response shape
export interface ProgramResponse {
  editable: EditableState;
  report: Report | null;
  last_synced: string | null;
  has_event_log?: boolean;  // True if domain_counts are derived from event log (real mode)
}

export interface ProgramSummary {
  slug: string;
  program_id: string | null;
}

export interface ProgramListResponse {
  programs: ProgramSummary[];
}

// Glossary
export interface GlossaryEntry {
  name: string;
  description: string;
  risk?: string;
}

export interface Glossary {
  A_STAR: number;
  domains: Record<string, GlossaryEntry>;
  alignment: Record<string, GlossaryEntry>;
  displacement: Record<string, GlossaryEntry>;
}

// App State
export type Theme = 'light' | 'dark' | 'system';
export type AppStatus = 'idle' | 'loading' | 'saving' | 'error';

