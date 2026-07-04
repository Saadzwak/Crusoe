# Prompt to paste into Claude Code

---

## Context (read this fully before doing anything — you have no prior memory of this project)

We are a team of 5 building a factory digital-twin system for the RAISE Summit 2026 hackathon, Crusoe track. The overall system has two layers, being built by two different people in parallel:

1. **Information layer** — already built by a teammate, in this same repo. It handles compression, reasoning, debate, security (HMAC), and observability/audit for telemetry that *will* come from a physics model. It currently runs on synthetic stand-in data because the physics model doesn't exist yet.
2. **Physical layer (this is what YOU are helping build)** — a set of **Multi-Head Physics-Informed Neural Networks (MH-PINN)**, one per monitored machine. Each PINN has a shared "body" (a recurrent structure, likely PI-LSTM) and several output "heads," one per physical phenomenon being monitored on that machine (e.g. one head for vibration, one for thermal/power behavior, one for degradation/remaining-life). The PINN's job is to understand a machine's real physical state from sensor data and eventually feed structured, HMAC-signed telemetry into the information layer described above.

**The narrative/demo framing**: the system is inspired by Michelin's tire manufacturing process. The specific machine we're focusing the physics on is the **tire curing press (vulcanization press)** — verified as the recognized bottleneck machine in tire manufacturing: real operating parameters we're using are temperatures up to 180°C, pressures exceeding 20 bar, cycle times of 10-15 minutes. No confidential Michelin data exists or is being used anywhere — we deliberately use public, verified, real datasets as physically-plausible stand-ins, and we are explicit about this being a stand-in, not real Michelin telemetry, whenever we present it.

**What THIS specific task is**: we do not yet know the exact shape/quality of the datasets we intend to train on. Before writing any PINN training code, we want you to download every dataset listed below, inspect its real structure, and produce a clear exploratory summary of each (columns, ranges, missing values, a few plots) — so we have a solid, verified starting point before designing the actual model architecture and training loop.

**This is exploration only. Do not build or train any model in this task. Do not touch, refactor, or interact with the existing information-layer code in this repo.**

---

## Git workflow — do this first, before downloading anything

1. Make sure the working tree is clean (check `git status`). If there are uncommitted changes that aren't yours, stop and ask.
2. Create and switch to a new branch dedicated to this work: `git checkout -b pinn-data-exploration`
3. Do all work for this task on this branch only. Do not merge into main/master yourself, and do not push directly to any branch a teammate is actively using. This branch exists specifically so this exploratory work doesn't collide with the teammate's information-layer commits.
4. Create a new top-level folder for this work: `pinn_data_exploration/` (separate from the existing information-layer folder(s) in this repo, so nothing gets mixed).

---

## Datasets to download

For each one below, try the download command given. **Some of these links I (the user, via a separate Claude conversation) personally tested and confirmed working just now; others I found via web search and believe are correct but have not personally tested the download in this exact form.** This is marked explicitly for each one — if a link fails, search for the current official page yourself before giving up on it, and tell me clearly which ones worked as-given vs which you had to adjust.

### 1. AI4I 2020 Predictive Maintenance Dataset — **personally verified working**
- Download: `https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip`
- It's a zip containing one file, `ai4i2020.csv`. Expect exactly 10,000 rows, 14 columns, 339 total rows with `Machine failure=1`, 115 with `HDF=1`, 95 with `PWF=1` — if your download doesn't match these numbers, something went wrong, stop and check.
- What it is: a synthetic-but-realistic dataset of milling-machine operation, with five independent failure modes (tool wear, heat dissipation, power, overstrain, random). We're using it as a stand-in for thermal/power/overstrain behavior on the curing press, because it has clean, documented, closed-form failure trigger rules (e.g. heat-dissipation failure triggers when (process temp − air temp) < 8.6 K AND rotational speed < 1380 rpm) — those closed-form rules are what a physics-informed head can be built around.

### 2. NASA C-MAPSS FD001 (Turbofan Engine Degradation Simulation) — **personally verified working**
- Download: `https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip`
- This is a **zip inside a zip**: extract the outer one, then extract `CMAPSSData.zip` found inside it.
- Focus on `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`. Expect exactly 20,631 rows in train, 13,096 in test, 100 in RUL — verify this after extraction.
- Format: space-separated, no header, 26 columns: `unit, cycle, op_setting_1, op_setting_2, op_setting_3, sensor_1...sensor_21`.
- Drop these columns first, they carry no signal in FD001 (constant values): `op_setting_3, sensor_1, sensor_5, sensor_10, sensor_16, sensor_18, sensor_19`.
- What it is: real-to-simulated run-to-failure trajectories of jet engines. We're using it as a stand-in for the "how much time is left before this machine fails" (Remaining Useful Life) head of the PINN — not because a curing press is a jet engine, but because the shape of the degradation trajectory (gradual sensor drift accelerating toward failure) is the pattern we want the model to learn to recognize and extrapolate.

### 3. CWRU Bearing Dataset — **found via search, not personally download-tested**
- Official page (may require navigating their interface, not a direct file link): `https://engineering.case.edu/bearingdatacenter/download-data-file`
- Alternative, likely easier to script against — Zenodo mirror with a permanent DOI: `https://zenodo.org/records/10987113` (try `https://zenodo.org/api/records/10987113/files` to get a machine-readable file list first)
- What it is: the world-reference bearing vibration dataset — accelerometer readings at drive-end, fan-end, and base, at 12kHz/48kHz, across 4 classes (healthy, inner-race fault, outer-race fault, ball fault). This is the vibration head of the PINN. The physics constraint we intend to use with it is the bearing fault characteristic frequency formulas (BPFO, BPFI, BSF) — you don't need to implement those yet, just get the raw data and understand its structure (file format, sampling rate, how fault type/size is encoded in filenames or metadata).

### 4. NASA IMS Bearing Dataset — **found via search, not personally download-tested**
- Download: `https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip`
- What it is: a **real run-to-failure** bearing dataset (not artificially induced faults like CWRU) — 4 bearings on one shaft, run until actual physical failure. Complements CWRU: CWRU teaches "what does this fault type look like," IMS teaches "what does the approach to failure look like over time." Also a vibration-head input.

### 5. NASA Milling Dataset — **found via search, not personally download-tested**
- Download: `https://phm-datasets.s3.amazonaws.com/NASA/3.+Milling.zip`
- What it is: cutting-tool wear data from a milling machine. Optional/secondary — only explore this one if the first four go smoothly and there's time left. It's a possible additional stand-in if we want a tool-wear-specific head later.

---

## What to actually produce for each dataset (this is the deliverable)

For each of the 5 datasets above, inside `pinn_data_exploration/<dataset_name>/`, produce:

1. A short `NOTES.md` describing: exact row/column counts you observed, column names and what each one physically represents (use the descriptions above, don't invent new ones), min/max/mean of each numeric column, count of missing values per column (should be 0 for all of these, confirm it), and anything that surprised you or didn't match what's described above.
2. At least one plot per dataset that shows what the data actually looks like over time or across the population (e.g. for C-MAPSS: a sensor value over operating cycles for a few engines; for AI4I: a scatter of two related columns colored by failure flag; for CWRU/IMS: a raw vibration waveform snippet and/or its frequency spectrum). Save plots as PNG files in the same folder.
3. Do not train anything. Do not build the PINN architecture yet. This task is exploration and verification only — the goal is that after this, we have full confidence in what each dataset actually contains before designing the model around it.

---

## When you're done

Summarize, in your final message, one paragraph per dataset: did it download and match the expected numbers, what does it actually look like, and anything I (the user) need to know or decide before we move to actually building the Multi-Head PINN. Commit your work on the `pinn-data-exploration` branch with a clear commit message. Do not merge to main.
