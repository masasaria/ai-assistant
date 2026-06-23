# Meeting Assistant

A local AI assistant that listens to your meeting audio in real time, transcribes questions, and answers them **based exclusively on a document you upload** — designed for thesis defenses, technical interviews, and presentations.

It can capture all of your computer's audio (including a meeting running inside a virtual machine) while you wear earphones, detect questions automatically, and answer hands-free.

## How it works

1. **Upload** your `.docx` or `.pdf` document (thesis, report, technical spec, etc.).
2. The app loads it and makes the **entire document** available to the answer engine.
3. **Start listening** — audio is transcribed live and questions are detected automatically.
4. Answers stream back in **2-3 seconds**, grounded strictly in your document — no outside knowledge, no invented facts.

The assistant also handles **clarification exchanges**: if the examiner asks a question, you clarify it, and they refine it, the app reads the whole back-and-forth and answers the final, refined question.

## ⚡ Recommended setup — fastest and most accurate

> **Transcription: Deepgram** · **Answers: Claude Sonnet 4.6**

This combination gives the best real-world experience:

- **Deepgram** transcribes audio in real time over a streaming WebSocket (`nova-2` model) — far lower latency than local Whisper, with excellent accuracy on technical terms.
- **Claude Sonnet 4.6** receives your **full document once and caches it on Anthropic's servers** (prompt caching). After the first question the document is "warm," so every subsequent answer reaches the model almost instantly — typically **2-3 seconds end to end**, complete and grounded in the document with no hallucination.

Local Whisper + Ollama still works fully offline and free, but expect slower responses and lower answer quality.

## Requirements

- **Python 3.10 or higher** — [python.org/downloads](https://python.org/downloads). During installation, check **"Add Python to PATH"**.
- **Windows 10 / 11 (64-bit)** — tested on Windows; works on macOS/Linux with minor adjustments.
- **A modern browser** — Chrome recommended (required for system-audio capture).

## Installation

Open a terminal (Command Prompt or PowerShell) and run:

```bash
pip install flask ollama faster-whisper python-docx pdfplumber anthropic numpy SpeechRecognition pymupdf
```

This may take 2-5 minutes. (Deepgram needs no Python package — it runs in the browser; you only provide an API key.)

## Answer engine — choose one

### Option A — Anthropic API / Claude (recommended)

1. Create an account at [console.anthropic.com](https://console.anthropic.com).
2. Go to **API Keys** and generate a key (starts with `sk-ant-api…`).
3. Paste the key directly in the app — no extra installation.

**Recommended model: `claude-sonnet-4-6`** — significantly better at reading a long document accurately, staying in persona, correcting wrong premises, and citing sources correctly than smaller models. The full document is cached, so cost per question stays low and answers stay fast.

### Option B — Local model with Ollama (offline, free)

1. Download and install Ollama from [ollama.com](https://ollama.com).
2. Pull a model: `ollama pull gemma4:12b` (~8 GB, once).
3. Start the server: `ollama serve` — keep this window open while using the app.

## Transcription engine — choose one

| Engine | Speed | Notes |
|---|---|---|
| **⚡ Deepgram** | **fastest** | Real-time streaming, very accurate. Needs a free API key from [deepgram.com](https://deepgram.com). **Recommended.** |
| **Whisper** | slower | Runs fully locally with `faster-whisper`. No internet needed. |
| **Google** | fast | Requires internet, no key. Lower accuracy on technical terms. |

## Running the app

1. Place `meeting_assistant_v3_app.py` in any folder.
2. Open a terminal in that folder and run:

   ```bash
   python meeting_assistant_v3_app.py
   ```

3. The browser opens automatically at `http://localhost:5000`.

## Usage

### Step 1 — Configure

- Select **Anthropic API** (recommended) or **Local (Ollama)**.
- If using Anthropic, paste your API key and choose **claude-sonnet-4-6**.
- Select transcription engine: **⚡ Deepgram** (recommended), **Whisper** (local), or **Google**. For Deepgram, paste its API key.
- Select audio source: **Mic** for your microphone, **System** to capture VM/meeting audio.

### Step 2 — Load document

- Click **Browse…** and upload your `.docx` or `.pdf`.
- Wait for processing to finish (a few seconds). On Anthropic, the document cache then warms in the background so your first question is fast too.

### Step 3 — Listen

- Select the language.
- Click **Start Listening**.
- For system audio: select your screen in Chrome's share dialog and tick **"Share system audio"**.
- Questions detected in the audio are transcribed and answered automatically.

**Hands-free:** say the trigger word **"responde"** (or use the **Answer now** button) to fire an answer without touching the keyboard — useful after a clarification exchange.

## Capturing meeting audio from a virtual machine

If your meeting runs inside a VM (VMware, VirtualBox, etc.) and you want the app to capture that audio on the host:

1. In the app, switch audio source to **🖥️ System**.
2. Click **Start Listening**.
3. In Chrome's share dialog, select the screen where the VM is running and enable **"Share system audio"**.

The app captures the host's system audio output, which includes whatever the VM plays through the host speakers. Your microphone can be captured at the same time so your clarifications are picked up too.

## Whisper model sizes (local mode only)

| Model | Speed | Accuracy |
|---|---|---|
| small | fastest | good |
| medium | balanced | better |
| large-v3-turbo | slower | best |

Loaded on `small` by default and cached. Switch models in the app (or the file).

## Troubleshooting

| Problem | Fix |
|---|---|
| `pip` not recognized | Reinstall Python and check "Add Python to PATH" |
| Microphone not working | Allow microphone access in the browser |
| Ollama not responding | Make sure `ollama serve` is running in another terminal |
| Deepgram auth failed | Check your Deepgram API key and internet connection |
| Transcription fails | Switch to Google or Whisper mode |
| Slow responses | Use Anthropic (Sonnet) + Deepgram; answers should be 2-3 s |
| No audio from system capture | Tick "Share system audio" in Chrome's share dialog |
| `[DocX] Loaded 0 bibliography entries` | Re-upload the document after restarting the app |

## Notes

- **Answers are based exclusively on the uploaded document** — the model is instructed never to use outside knowledge and to say when something isn't in the document instead of inventing it. It will also correct a wrong premise in a question.
- Answers are kept **concise** — straight to the point, no filler — and arrive in **2-3 seconds** with the recommended setup.
- On Anthropic, the **full document** is cached on Claude's servers via prompt caching, so it's sent once and reused cheaply for every question (kept warm automatically during your session).
- Word documents using the built-in citation manager (IEEE, APA, etc.) are fully supported — author names are extracted from the citation XML and injected automatically.
- **Privacy:** with Whisper + Ollama, nothing leaves your machine. With Anthropic and/or Deepgram, document text and audio are sent to those services to generate transcripts and answers. If confidentiality is critical, use the fully local setup.
