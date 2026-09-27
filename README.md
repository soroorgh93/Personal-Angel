# PersonalAngel

**An on-device multimodal safety investigator.** PersonalAngel watches a camera or listens to a microphone, turns what it
sees and hears into structured events, investigates the most likely incident with a local vision-language model, argues
against its own conclusion, and recommends a safe, policy-gated action. Everything runs on one computer. No video,
audio, transcript or memory ever leaves the device.

> It does not classify a recording. It investigates it.

## What it does

| Stage | What happens | Models |
|---|---|---|
| Perceive | People, objects, weapons, body pose, scene type, speech in 13 languages, sounds, text risk | YOLO11, ByteTrack, YOLO11-pose, CLIP, Faster-Whisper, CLAP, Detoxify |
| Structure | Time-stamped events: who, did what, to what, where, with what confidence and severity | rule engines over the perception outputs |
| Rank | An event graph (events, evidence, people, places) ranked with PageRank picks the frames worth showing to the big model | NetworkX |
| Investigate | A ReAct master agent with eleven costed tools and a compute budget tests the primary hypothesis | Qwen2.5-VL-7B (any OpenAI-compatible vision model) |
| Verify | The same model, prompted with the opposite job, must fail to find the innocent explanation | critic role |
| Decide | A transparent rule: act when belief × harm ≥ cost of a wrong action; ask the person only when it is safe to ask | policy gate |
| Act | Simulated dispatch (911, police with location, security, family, owner, advise, log) written to an audit log | executor |

Event kinds: fall, person down, slumped and unresponsive, weapon visible, aggressive interaction (fight), threatening,
hateful, abusive and distress speech, infant distress, alarming sound (gunshot, scream, glass), object removed, scene
change, normal activity.

## Quick start

Requirements: Python 3.10–3.12, ffmpeg, a GPU is optional (the CPU profile uses a 4B model on Ollama).

```bash
git clone <your repository url> personal-angel && cd personal-angel
python -m venv .venv && source .venv/bin/activate          # Windows: .\.venv\Scripts\activate
pip install -r requirements/base.txt -r requirements/vision.txt -r requirements/audio.txt
pip install -e . --no-deps
python scripts/download_models.py                            # YOLO11, pose, weapon and fall weights, CLIP, Whisper
python scripts/fetch_real_demo_clips.py --out data/demo      # public example recordings with sources and licences
python -m pytest -q                                          # 35 tests, fixture profile, no GPU needed
```

Serve a local model, then start the console:

```bash
# CPU laptop: Ollama
ollama pull qwen3.5:4b
python -m personal_angel serve --profile pc_cpu --port 8600

# GPU machine (tested on a workstation with 128 GB unified memory): vLLM
bash scripts/launch_vllm.sh                                  # Qwen/Qwen2.5-VL-7B-Instruct on http://127.0.0.1:8000/v1
python -m personal_angel serve --profile workstation --port 8600
```

Open http://127.0.0.1:8600. Drop a recording (or pick an example), press **Investigate**, and watch the trace:
scene → events → evidence → hypothesis → tool calls → critic → policy gate → verdict → report and cost telemetry.
The **Live** panel records 10–30 s from the browser's camera and microphone and investigates it the same way.

### Windows app (one click)

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1
```

The setup creates a virtual environment, downloads the models and example recordings, runs the tests and builds
`PersonalAngel.exe` with a Desktop shortcut. Install [Ollama](https://ollama.com/download) first so the setup can pull
the local model. Double-click the shortcut: the app opens in its own window. In the **Live** panel choose
**Camera + mic**, allow the camera, act a short scene, and the app investigates it.

Linux GPU workstations: `bash scripts/setup_linux.sh` then `WEB=1 bash scripts/run_linux.sh`.

Command line:

```bash
python -m personal_angel analyze data/demo/urfd_fall_01.mp4 --profile workstation --answer ""
python scripts/benchmark.py --profile workstation --manifest data/eval/real_cases.jsonl --variants full no_critic no_pagerank
```

## Configuration

`config/base.yaml` holds every knob; `pc_cpu.yaml`, `pc_gpu.yaml`, `workstation.yaml` and `fixture.yaml` override it.
Environment variables `ANGEL_<SECTION>__<KEY>` override a single value (`ANGEL_LLM__BASE_URL=http://127.0.0.1:8000/v1`).

Key settings:

| Setting | Default | Meaning |
|---|---|---|
| `policy.escalate_threshold` | 0.70 | belief needed for police / 911 |
| `policy.ask_threshold` | 0.40 | belief needed to ask the person |
| `policy.action_cost` | police 0.55, 911 0.35, reroute 0.25, parents 0.20, security 0.15, owner 0.10, ask 0.05 | cost of a wrong action |
| `agent.compute_budget` | 100 (CPU) / 160 (GPU) | budget in compute units; each tool has a cost |
| `agent.planner` | `hybrid` (Fast, 3 model calls) or `llm` (Deep, the model plans every step) | who plans |
| `agent.ask_timeout_s` | 20 | how long to wait for the person's answer |
| `llm.max_images_per_call` | 3 (CPU) / 8 (GPU) | evidence frames per vision call |
| `memory.policy_docs` | three example policies in `examples/policies/` | plain-English site rules retrieved by the agent |

## Belief and decision rule

```
p0  = clip(0.10 + 0.75 · detector_confidence, 0.05, 0.90)
p'  = sigmoid( logit(p) + 2 · w · support )        support ∈ [−1, 1], w = model confidence clipped to [0.4, 1]
act(a) is justified  ⇔  p ≥ threshold(a)  and  p · severity ≥ (1 − p) · cost(a)
```

Belief floors protect established evidence (weapon + second opinion + gunshot ≥ 0.75; weapon + second opinion ≥ 0.45;
group fight ≥ 0.50). Weapons, fights, threats and infants are never asked a question; silence after a direct question
counts as evidence.

## Layout

```
personal_angel/
  perception/   detector, pose, scene (CLIP), audio (Whisper, CLAP, text risk), pipeline, model registry
  events/       event builder, event graph + PageRank
  agent/        master agent (ReAct), tools, critic, policy gate, safety guard, actions, LLM client
  memory/       episodic / procedural memory (SQLite), semantic cache
  server/       Starlette app, server-sent events, upload normalisation
  telemetry.py  tokens, model calls, GPU, energy, cloud-equivalent cost, edge/cloud placement
config/         profiles        ui/  operator console        scripts/  setup, model download, benchmark, training
examples/       site policies   tests/  35 tests             data/eval/  labelled case manifests
```

## Measured (same clip, same code, same three model calls)

| | laptop CPU (4B model, Ollama) | GPU workstation (7B vision model, vLLM) |
|---|---|---|
| wall time, 5 s bar-fight clip | 195.4 s | 53.4 s |
| generation speed | 3.3 tok/s | 13.3 tok/s |
| energy | 0.0076 kWh | 0.0021 kWh |
| verdict | CLEAR (missed the fight) | ALERT (correct) |

## Safety and privacy

All inference is local. Dispatch is simulated in this build: every alert is built as a real object and written to
`runs/<id>/actions.jsonl`, not sent. Transcripts and documents are untrusted input: sanitised, screened for injected
instructions and quoted, never executed. The system does not identify individuals and stores no face embeddings.
Deployments must be opt-in, signposted and compliant with local law on recording.

## Datasets and models

UR Fall Detection (Kwolek & Kępski 2014, CC BY-NC-SA, research only); RWF-2000 (Cheng, Cai & Li 2020); Surveillance
Camera Fight Dataset (Aktı et al. 2019); ESC-50 (Piczak 2015); Multilingual HateCheck (Röttger et al. 2022); Pexels
stock clips. Models: Ultralytics YOLO11 / YOLO11-pose; public gun-knife and fallen-person YOLO weights; OpenAI CLIP
ViT-B/32; Systran Faster-Whisper; LAION CLAP; Unitary Detoxify; Qwen2.5-VL-7B-Instruct. All pretrained; none were
trained here. Training scripts for a weapon detector, a fall model and a triage classifier are in `scripts/` as a
starting point.

## Licence

Apache-2.0 (see `LICENSE`).
