# Non‑Semantic Mechanistic Interpretability Rubric (Core)

## Purpose

Score proposals and results for geometry‑ and dynamics‑based interpretability, without semantic or anecdotal shortcuts.

---

## Scoring Scale

Each criterion is scored **0–3**:

- **0** – Fails; do not proceed.
- **1** – Weak; major revision needed.
- **2** – Solid; minor gaps acceptable.
- **3** – Strong; clearly advances understanding.

**Acceptance:**  
No criterion = 0.  
Total score ≥ 18/27 (≈67%).

---

## Criteria

### 1. External Grounding

**1A. Methodological grounding**

- 0 – Uses ad‑hoc metrics with no reference to established mathematics or prior work.  
- 1 – Vaguely inspired by prior work, but key steps are invented and untested.  
- 2 – Clearly based on known mathematical or algorithmic ideas.  
- 3 – Closely follows a well‑defined method and makes comparison possible.

**1B. Publishability**

- 0 – Would be dismissed as anecdotal or arbitrary.  
- 1 – Interesting but under‑specified or poorly controlled.  
- 2 – Could be shared as a serious technical note.  
- 3 – Has the rigor and clarity needed for a strong research venue.

---

### 2. Theoretical Clarity

**2A. Specific structural claim**

- 0 – No clear claim; only “let’s see what happens.”  
- 1 – Broad story, but test would look similar if the story were wrong.  
- 2 – Tests a specific structural or dynamical property with a clear outcome.  
- 3 – Tests several related properties that jointly constrain or falsify the theory.

**2B. No self‑fulfilling design**

- 0 – Builds the object it later claims to find.  
- 1 – Significant circularity; outcome is largely baked into setup.  
- 2 – Setup allows the claim to be wrong in principle.  
- 3 – Includes an explicit null or baseline that can contradict the theory.

---

### 3. Non‑Semantic Discipline

**3A. Semantic independence**

- 0 – Relies on labels, meanings, or task scores as primary signals.  
- 1 – Mostly geometric/dynamic, but uses meaning‑based criteria in core steps.  
- 2 – Core measurements depend only on representation structure or dynamics.  
- 3 – Method is provably invariant to relabeling of inputs or outputs.

**3B. Structural insight**

- 0 – Produces numbers or plots with no clear structural message.  
- 1 – Some patterns, but unclear how they relate to model internals.  
- 2 – Reveals where and how important transformations happen.  
- 3 – Gives a coherent picture of the model’s internal organization.

---

### 4. High‑Stakes Value

**4A. Scale and realism**

- 0 – Pure toy with no path to real models.  
- 1 – Very small pilot; unclear if it scales.  
- 2 – Applied to a substantial real model.  
- 3 – Either multiple models or a clear scaling argument.

**4B. Completeness**

- 0 – One narrow case; no general conclusion possible.  
- 1 – Limited coverage; many unresolved caveats.  
- 2 – Covers several layers/conditions; conclusions are reasonably stable.  
- 3 – As thorough as budget allows; includes basic sensitivity checks.

**4C. Strength of evidence**

- 0 – Anecdotal; no statistics or baselines.  
- 1 – Some quantification, but weak or noisy.  
- 2 – Clear effect with appropriate tests and reproducibility.  
- 3 – Effect is robust under controls; would convince a skeptic.

---

### 5. Compute Realism

**5A. Feasibility on current hardware**

- 0 – Exceeds available resources.  
- 1 – Barely feasible; high risk of failure.  
- 2 – Fits comfortably in available budget.  
- 3 – Lightweight; allows fast iteration.

**5B. Strategic use of compute**

- 0 – Trivializes the question to save compute.  
- 1 – Over‑cuts; remaining test is marginal.  
- 2 – Uses focused subsets (layers, samples, projections) without losing the point.  
- 3 – Turns constraints into advantages (e.g. careful layer choice, efficient estimators).

---

## Using the Rubric

- **Before coding**: score the idea; reject or revise if any 0 appears or total < 18.  
- **After running**: score the result; treat low‑scoring work as exploratory only.  
- **When choosing between ideas**: prefer higher total; break ties by 4A–4C first.

Code Style:
- Use only one script, one run, no flags for all.
- do not save in json or other ways results.
- use maximum 5 component seperators (whether - or =) so we wont waste tokens.
- do not add assumptions in docstrings or comments in the code or its prints. Only print meaningful results we can interpret and analyse.
- do add context in docstrings for future reference.
- always use the real Atlas and Router instead of toys.
- prefer to use real Olmo tokenizer, and parameters. 
- use bfloat16 whenever useful so we wont have OOMs.

Our PC Setup:
# Mini PC Specifications - ARB19D-P08-CH

## System Overview

**System Type:** x64-based PC  
**System Family:** MiniBox  

---

## Branding & Manufacturer Information

**Retail Brand:** TexHoo  
**OEM Manufacturer:** ZNRS (SIXUNITED)  
**System Manufacturer:** SU  
**System Model:** UM660  
**System SKU:** UM660  
**Baseboard Manufacturer:** SU  
**Baseboard Product:** ARB19D  
**Baseboard Version:** Version 1.0  
**Baseboard Serial Number:** ARB19D314B06H0350  
**Product ID:** ARB19D-P08-CH  

---

## Physical Specifications

**Product Style:** Mini PC  
**Material:** Plastic  
**Dimensions:** 130 × 127 × 45.1 mm  
**Weight:** ~470g  
**Kensington Lock:** Yes  

---

## BIOS Information

**BIOS Manufacturer:** American Megatrends International, LLC. (AMI)  
**BIOS Version:** ALASKA - 1072009  
**BIOS Date:** October 24, 2024  
**SMBIOS Version:** 3.4  
**BIOS Version (SMBIOS):** 104  
**System BIOS Version:** 5.24  
**BIOS Customization:** SIXUNITED customization  
**Embedded Controller:** Version 1.14  

---

## Processor (CPU)

**Manufacturer:** AMD (DirectAMD)  
**Model:** AMD Ryzen 5 6600H with Radeon Graphics  
**Architecture:** AMD64 Family 25 Model 68 Stepping 1  
**CPU Type:** AMD Rembrandt/Phoenix/Hawk Point  
**Cores:** 6 physical cores  
**Logical Processors:** 12 (6 cores × 2 threads)  
**Base Clock Speed:** 3.30 GHz  
**Max Clock Speed:** 3.30 GHz  
**L2 Cache:** 3 MB  
**L3 Cache:** 16 MB  
**Socket:** FP7r2  
**Virtualization:** Enabled  

---

## Graphics (GPU)

**GPU Type:** AMD Integrated Graphics  
**GPU Model:** AMD Radeon(TM) Graphics  
**GPU Processor:** AMD Radeon Graphics Processor (0x1681)  
**Video Architecture:** PCI  
**Adapter RAM:** 4,293,918,720 bytes (~4 GB shared system memory)  
**DAC Type:** Internal DAC (400MHz)  
**Current Driver Version:** 32.0.12019.1028  
**Driver Date:** October 11, 2024  
**Driver INF:** u0408380.inf  
**Driver Location:** C:\WINDOWS\System32\DriverStore\FileRepository\u0408380.inf_amd64_a2a50bd2d2429936\  

**Known Issue:** HDMI VideoOutputTechnology showing as invalid (4294967295)  
**Recommended Driver:** ZNRS_Drive(6600H)_2024.11 package (October 2024)  

---

## Memory (RAM)

**Total Physical Memory:** 25,499,254,784 bytes (~24 GB)  
**Memory Type:** DDR5 SO-DIMM  
**Memory Configuration:** Dual channel  
**Memory Slots:** SO-DIMM DDR5 slot  
**Memory Capacity Range:** 4GB up to 64GB  

**Installed Memory Modules:**
- Module 1: A-DATA Technology, 16 GB, 4800 MHz, Part Number: CBDAD5S480016G-B
- Module 2: A-DATA Technology, 16 GB, 4800 MHz, Part Number: CBDAD5S480016G-B

**Total:** 32 GB DDR5-4800 (2 × 16 GB modules)

---

## Storage

**Storage Interface:** 2 × M.2 2280 PCIe 4.0 × 4 SSD slots  

**Installed Storage:**
- Primary SSD: Lexar SSD NQ6A1 512GB (512,105,932,800 bytes = ~512 GB)
- External: PHILIPS USB Device (31,453,470,720 bytes = ~30 GB)

---

## External I/O Ports

### Front Panel
- **Audio Jack:** 1 × Ø3.5mm Combo (Headphone/Microphone)
- **USB Type-A:** 2 × USB 3.2 Gen2
- **USB Type-C/Thunderbolt 4:** 1 × USB4

### Rear Panel
- **DC Power Jack:** 1 × DC Input
- **HDMI:** 1 × Standard HDMI 2.0 (4K@60Hz) ⚠️ *Currently experiencing "No Signal" issue*
- **DisplayPort:** 1 × DP (4K@60Hz)
- **USB Type-A:** 1 × USB 3.2 Gen2 + 1 × USB 2.0
- **Ethernet (RJ45):** 2 × 2.5G LAN ports

### Indicators
- **Power LED:** Blue LED
- **Power Key:** On Front Panel

---

## Network Connectivity

### Wireless
- **WiFi Chip:** M.2 2230 PCIe & USB
- **WiFi Standard:** 802.11b/g/n/ac/ax 2×2
- **WiFi Frequency:** 2.4 GHz / 5 GHz
- **Bluetooth Version:** 5.0

### Wired
- **Ethernet:** 2 × Gigabit LAN (2.5G capable)
- **LTE:** 4G LTE (Not Available/NA)

### Power Management
- **Modern Standby:** Supported

---

## Operating System

**OS:** Windows 11  
**Language Support:** Multi-language  
**Alternative OS:** Linux (supported)  

---