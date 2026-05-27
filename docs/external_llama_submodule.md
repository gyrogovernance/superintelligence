# external/llama.cpp (git submodule)

`external/llama.cpp` is a **submodule**: this repo stores only a commit pointer, not the full llama.cpp tree.

| What | Where |
|------|--------|
| Source code | https://github.com/basilkorompilias/llama.cpp (`main`) |
| Pointer in superintelligence | `external/llama.cpp` (one line in git: commit SHA) |
| Local backup (not tracked) | `external/llama.cpp_backup/` (gitignored) |

## Clone superintelligence with llama

```powershell
git clone --recurse-submodules <superintelligence-url>
cd superintelligence
```

If you already cloned without submodules:

```powershell
git submodule update --init external/llama.cpp
```

## Update llama.cpp

Work inside the submodule, commit and push **on the llama repo**:

```powershell
cd external/llama.cpp
git checkout main
git pull origin main
# edit, then:
git add -A
git commit -m "your message"
git push origin main
cd ../..
git add external/llama.cpp
git commit -m "Bump llama.cpp submodule"
```

## Build

From repo root (see `src/tools/gyroscopic/helpers/build_llama_cpp_windows.ps1`):

```powershell
powershell -File src/tools/gyroscopic/helpers/build_llama_cpp_windows.ps1
```

Binary: `external/llama.cpp/build/bin/Release/llama-cli.exe`

## Model

Download Bonsai GGUF (gitignored under `data/models/`):

```powershell
python scripts/download_bonsai.py
```

Config: `config/gyroscopic_llm.yaml`
