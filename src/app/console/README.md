# AIR Governance Console

A browser-based UI for managing AIR program contracts.

## Architecture

The console is a **thin view layer** over existing AIR logic:

1. **Backend** (`api/server.py`): FastAPI server wrapping `src/app/cli/store.py`
2. **Frontend** (`ui/`): React + Vite + Tailwind application

## Quick Start

### First-Time Setup

Install dependencies:

```bash
python air_installer.py
```

This checks prerequisites (Python, Node.js, npm) and installs frontend dependencies.

### Run Both Servers (Recommended)

From the program root:

```bash
python air_console.py
```

This starts both the backend (port 8000) and frontend (port 5173) servers. Press Ctrl+C to stop both.

### Run Servers Separately

**Backend:**

```bash
cd src/app/console/api
uvicorn server:app --reload --port 8000
```

**Frontend:**

```bash
cd src/app/console/ui
npm install
npm run dev
```

The frontend dev server proxies `/api` requests to `http://localhost:8000`.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/programs` | List all programs |
| POST | `/api/programs` | Create new program |
| GET | `/api/programs/{slug}` | Get program details and report |
| PUT | `/api/programs/{slug}` | Update program counts |
| DELETE | `/api/programs/{slug}` | Delete program and artifacts |
| GET | `/api/programs/{slug}/bundle` | Download verified bundle |
| GET | `/api/glossary` | Get terminology glossary |

## Features

- Create, edit, and delete AIR programs
- Automatic sync on edit (500ms debounce)
- Domain distribution visualisation
- Alignment/displacement incident counters
- Balance gauges showing aperture deviation from A*
- Kernel state display
- Bundle download for verified artifacts
- Dark/light/system theme support
- Glossary modal for terminology
- Responsive design

## Program Structure

```
src/app/console/
  api/
    __init__.py
    server.py         # FastAPI backend
  ui/
    package.json
    vite.config.ts
    tailwind.config.js
    src/
      main.tsx        # Entry point
      App.tsx         # Main application
      api.ts          # API client
      types.ts        # TypeScript types
      theme.ts        # Theme utilities
      index.css       # Tailwind styles
      components/
        Header.tsx
        Stepper.tsx
        BalancePanel.tsx
        DomainsPanel.tsx
        PrinciplesPanel.tsx
        KernelPanel.tsx
        ReportPanel.tsx
        GlossaryModal.tsx
        ConfirmModal.tsx
```

## Production Build

```bash
cd src/app/console/ui
npm run build
```

Output goes to `dist/`. The backend can serve these as static files.

