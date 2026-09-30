# Voice to Text (English + Indian languages, CPU)

Offline speech-to-text that runs on a regular CPU on **macOS, Windows or Linux**. No GPU
and no ffmpeg install are needed. Speak **English, Hindi, Marathi or another Indian language**
and get the text back **in English**.

The server detects the spoken language and picks the best model for it:

| You speak | What happens | Models |
|---|---|---|
| English | Transcribed as is | Whisper `large-v3-turbo` ([faster-whisper](https://github.com/SYSTRAN/faster-whisper)) |
| Hindi, Marathi, Bengali, Gujarati, Punjabi, Tamil, Telugu, Kannada, Malayalam, Odia, Assamese, Nepali (auto-detected) or 10 more Indian languages (pick them in the dropdown) | Transcribed in the spoken language, then translated to English | AI4Bharat [IndicConformer](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual) → [IndicTrans2](https://huggingface.co/ai4bharat/indictrans2-indic-en-dist-200M) |
| Any other language (Whisper detects it, or pick it in the API) | Transcribed, not translated | Whisper |

Language detection, in order:

1. **English or not** (~50 ms): a small classifier on [VoxLingua107](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa)
   voice features, trained on real Indian-accented English (`english_detector.json`). When it's
   unsure (~5% of clips), or the speech doesn't sound like any auto-detect language, Whisper's own
   language ID decides.
2. **Which Indian language**: VoxLingua107 combined with IndicConformer's own scores, plus:
   - **Language memory**: the server learns which languages *you* speak and leans towards them
     (no setup; see [How it learns your languages](#how-it-learns-your-languages)).
   - **Word evidence**: Hindi vs Marathi is tipped by the words used (है/को/क्या vs आहे/काय/-ाला).
3. **Safety net**: if the Indian-language model wrote English words out phonetically
   (e.g. "वॉट आर माई ऑर्डर्स"), or heard no words at all, the clip is redone with Whisper.

How fast (on a recent laptop CPU): Hindi/Marathi and other Indian languages **~0.1–0.5 s**, English
**~2.5 s** per request. About 1 clip in 20 takes ~2 s longer while Whisper double-checks the language.

What's included:

| File | What it is |
|---|---|
| `server.py` | REST API (FastAPI). Also serves the web UI |
| `indic.py` | Indian-language pipeline: language detection, speech recognition, translation |
| `english_detector.json` | Small trained English-vs-Indian classifier used by `indic.py` (3 KB) |
| `index.html` | Chat-style voice UI: tap the mic, speak, and get the English text back |
| `transcribe.py` | Command-line tool (Whisper only) for transcribing audio files |
| `requirements.txt` | Python dependencies |

### Why two pipelines

Measured on real recordings (Google FLEURS test set) on this project's CPU setup:

| | Hindi | Marathi |
|---|---|---|
| Word error rate, Whisper `large-v3-turbo` (language forced) | 30.5% | 82.8% |
| Word error rate, **IndicConformer** | **11.6%** | **17.2%** |
| Speech → English BLEU, Whisper `large-v3` translate | 15.3 | 9.8 |
| Speech → English BLEU, **IndicConformer → IndicTrans2** | **30.6** | **30.6** |

Time for a 10 s clip: English (Whisper) ~2.5 s; Indian languages ~0.1–0.5 s.

Routing, on speakers not used for tuning: real Indian-accented English went to the English path
100% of the time (EdAcc corpus, 548 clips; the old detector managed only 23–79%), and Indian-language
speech went to the Indian path 99–100% of the time. The specific Indian language was right for 98%
of full sentences across 12 languages. Very short fragments (~3 s) are harder; the language memory
and word evidence raise a Hindi/Marathi speaker from 89% to 92% in simulation.

---

## Quick start

You need **Python 3.10–3.13** ([python.org](https://www.python.org/downloads/)), about
**6 GB of free RAM**, and an internet connection for the first run. The models download once
(~5.3 GB: Whisper 1.6 GB, Indian-language models 3.7 GB).

English only, or a smaller machine? Start with `INDIC=0` to skip the Indian-language
models (~2 GB of RAM). See [Configuration](#configuration).

**Linux:** install the CPU build of PyTorch first, or pip downloads several GB of CUDA libraries:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`

### macOS / Linux

```bash
cd speech-to-text
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python server.py
```

### Windows (Command Prompt)

```bat
cd speech-to-text
py -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
python server.py
```

### Windows (PowerShell)

```powershell
cd speech-to-text
py -m venv .venv
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned   # one time only, allows the activate script
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python server.py
```

### Then open the app

Go to **http://127.0.0.1:8000** in Chrome, Edge, Firefox or Safari.

1. Wait until the top-right status shows a green dot and **Ready**.
   On the first run this takes several minutes while the models download.
2. Tap the **microphone** and allow microphone access when the browser asks.
3. Speak in English, Hindi, Marathi or another Indian language, then tap the microphone again.
4. Your voice note appears on the right. The reply shows the **English** text, with what you
   said in its original script underneath, and a label such as *Marathi → English*.

Leave the dropdown on **Auto-detect**, or pick the language you'll speak (slightly more
reliable, and needed for the 10 Indian languages that aren't auto-detected). Translation is
automatic: Indian-language speech always comes back in English, and English stays as it is.
(API users who want the untranslated transcript can send `output=original`, see [REST API](#rest-api).)

---

## Testing it

1. Start the server (`python server.py`) and open **http://127.0.0.1:8000**. Leave the dropdown
   on **Auto-detect**.
2. Say these and check the reply and the label under it:

   | Say | Expected reply | Label |
   |---|---|---|
   | "What are my orders for today" | What are my orders for today? | **English** (no Devanagari anywhere) |
   | "मेरे आज के ऑर्डर क्या हैं" | English translation, Hindi underneath | **Hindi → English** |
   | "आज माझ्या ऑर्डर्स काय आहेत" | English translation, Marathi underneath | **Marathi → English** |
   | "कस्टमर को वापस कॉल करो" | Call the customer back | **Hindi → English** |
   | "ग्राहकाला परत फोन करा" | Call the customer back | **Marathi → English** |

   The time in the label should be ~0.1–0.5 s for Hindi/Marathi and ~2–3 s for English.
3. Watch the terminal where the server runs. Every transcription prints one line showing how it was routed:

   ```
   [indic] Hindi -> English in 0.19s: 'what are my orders for today'
   [indic] Marathi -> English in 0.29s: 'call the customer back'
   [whisper] English in 2.51s: 'What are my orders for today?'
   ```

   `[indic]` means the Indian-language pipeline handled it, `[whisper]` means Whisper did, and
   `[no speech]` means nothing was heard. Requests that fail with an error print nothing.
   A Hindi or Marathi phrase that prints `[whisper]` was mistaken for English (see
   [Troubleshooting](#troubleshooting)).
4. After 2–3 Hindi and Marathi sentences, open **http://127.0.0.1:8000/api/health**:

   ```json
   "learned_languages": ["hi", "mr"],
   "language_memory": {"mr": 2.91, "hi": 1.8}
   ```

   `language_memory` counts how much each language has been heard; a language joins
   `learned_languages` after about two confident uses (or one dropdown pick). See
   [How it learns your languages](#how-it-learns-your-languages).
5. Test a file from the terminal (in Windows PowerShell type `curl.exe`):

   ```bash
   curl -F "audio=@recording.m4a" http://127.0.0.1:8000/api/transcribe
   ```

   The response shows `engine` (`indic` or `whisper`), `language`, `text` (English) and
   `transcript` (original script).

---

## Running it day to day

Setup is one-time only. After that, from the `speech-to-text` folder:

| OS | Start the server |
|---|---|
| macOS / Linux | `source .venv/bin/activate` then `python server.py` |
| Windows cmd | `.venv\Scripts\activate.bat` then `python server.py` |
| Windows PowerShell | `.venv\Scripts\Activate.ps1` then `python server.py` |
| Any OS, without activating | `.venv/bin/python server.py` (macOS/Linux) or `.venv\Scripts\python server.py` (Windows) |

The server prints something like:

```
Loading Whisper 'large-v3-turbo' (CPU, int8)... first run downloads ~1.6 GB
Voice chat UI: http://127.0.0.1:8000
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
Whisper ready.
Loading Indian-language models... first run downloads ~3.7 GB
Indian-language models ready.
[indic] Hindi -> English in 0.19s: 'what are my orders for today'     <- one line per request
```

**Check that it's running:**

```bash
curl http://127.0.0.1:8000/api/health
```

→ `{"status":"ok","model":"large-v3-turbo","components":{"whisper":"ok","indic":"ok"},"learned_languages":[],"language_memory":{}}`
(`"loading"` means the models aren't ready yet. `learned_languages` and `language_memory` appear once
the Indian-language models are loaded, and fill up as you speak.)

Or just open http://127.0.0.1:8000/api/health in a browser.

**Stop it:** press `Ctrl+C` in the terminal where it's running.

**Use a different port** if 8000 is taken:

| OS | Command |
|---|---|
| macOS / Linux | `PORT=8001 python server.py` |
| Windows cmd | `set PORT=8001` then `python server.py` |
| Windows PowerShell | `$env:PORT="8001"; python server.py` |

Other settings (model, limits and so on) use environment variables the same way. See
[Configuration](#configuration).

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Every transcription fails with *Could not decode audio*, or the CLI shows `unexpected keyword argument 'metadata_errors'` | PyAV 19 is installed. Run `pip install "av<19"` (requirements.txt already pins this) |
| `pip install` fails with *No matching distribution found* | Your Python version isn't supported on this machine. Install Python 3.12 or 3.13. See [Platform notes](#platform-notes) |
| PowerShell says *running scripts is disabled* | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use Command Prompt instead |
| *Address already in use* / *WinError 10048* | Another program (or another copy of the server) is using port 8000. Stop it, or start with a different `PORT` (see above) |
| Page shows **API offline** | The server isn't running, or you opened a different address. Start `python server.py` and use http://127.0.0.1:8000 |
| Page stays on **Loading models…** | First run: the models (~5.3 GB) are still downloading. Watch the terminal. It needs internet access to huggingface.co |
| Page shows **Model failed to load** | Read the message in the terminal. Usually no internet on first run, or a misspelled `WHISPER_MODEL` |
| Page shows **Ready · English only** | Either `INDIC=0` is set, or the Indian-language models failed to load (message in the terminal and in the page's hint). English still works. Usually no internet on first run, not enough RAM, or an Intel Mac (English only) |
| The computer slows down or the server is killed | Not enough RAM: everything needs ~6 GB. Close other apps, or run English only with `INDIC=0` |
| Hindi/Marathi comes out wrong or in the wrong language | If a phrase is mistaken for a language you never speak (e.g. Nepali, Punjabi), that improves by itself after a few sentences (language memory). Hindi vs Marathi relies on the words used; the memory can't separate two languages you both speak. Very short clips (under ~3 s) are the hardest. Picking the language in the dropdown always works |
| English comes out in Devanagari / as Hindi | Shouldn't happen with this version; the old Whisper-only version did this for Indian-accented English. Make sure you restarted `server.py`. If it still happens, pick **English** in the dropdown. To report it, save the recording (Chrome/Edge: ⋮ on the audio player → Download; Firefox/Safari: right-click the player) and send it to whoever maintains `english_detector.json`, so it can be retrained on your voice |
| Hindi/Marathi prints `[whisper] English` in the terminal | The English detector mistook it for English. Pick the language in the dropdown for now, and save the recording to report it (see the row above) |
| `learned_languages` stays empty | It only fills from Hindi/Marathi (or other Indian-language) speech: English isn't tracked. Check the terminal lines say `[indic] ...`. A language needs about two confident uses; `language_memory` shows the progress. If both fields are missing from `/api/health`, the Indian-language models aren't loaded (`INDIC=0`, still loading, or failed) |
| Deleted `language_memory.json` but it came back | The running server keeps the memory in RAM and writes it back. Stop the server, delete the file, then start it again |
| *Microphone permission denied* | Allow the microphone for the site in your browser. Also check the OS setting: **macOS** System Settings → Privacy & Security → Microphone (enable your browser); **Windows** Settings → Privacy & security → Microphone |
| Mic doesn't work from a phone or another computer | Browsers only allow the microphone on `localhost` or over HTTPS. See [Access from other devices](#access-from-other-devices) |
| English transcription is slow | Use a smaller Whisper model: `WHISPER_MODEL=small python server.py` (see [Choosing a model](#choosing-a-model)). Indian languages are already fast (~0.1–0.5 s) |
| *(No speech detected)* | The recording was silent or too quiet. Check the right microphone is selected in your browser or OS |
| `403 Cross-origin request blocked` | You're calling the API from a web page on another address. See `CORS_ORIGINS` in [Configuration](#configuration) |
| `400 Invalid host header` | You opened the server by an IP or name it doesn't accept. Add it to `ALLOWED_HOSTS` |

---

## Command-line tool

`transcribe.py` uses Whisper only (no Indian-language pipeline); use the server for Hindi,
Marathi and other Indian languages. With the virtual environment active:

```bash
python transcribe.py meeting.mp3                        # print the transcript
python transcribe.py meeting.mp3 --output meeting.srt   # save subtitles
python transcribe.py meeting.mp3 --output meeting.txt   # save plain text
python transcribe.py call.wav --language hi             # force a language (Hindi)
python transcribe.py call.wav --model small             # faster, less accurate
```

It prints a "real-time factor" at the end (processing time ÷ audio length). Below 1.0 means
faster than real time.

---

## REST API

With the server running, interactive docs are at **http://127.0.0.1:8000/docs**.

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/transcribe` | Transcribe an audio file |
| `GET` | `/api/health` | `status` (`ok` / `loading` / `error`), `model`, `components` (`whisper`, `indic`: `ok` / `loading` / `error` / `disabled`), `learned_languages` and `language_memory` (only when the Indian-language pipeline is loaded; see [How it learns your languages](#how-it-learns-your-languages)), plus `detail` on errors |
| `GET` | `/api/languages` | `languages`: `[{"code", "name", "auto", "translated"}]` for English and the 22 Indian languages (English only when the Indian-language pipeline is off or not loaded); `auto` = auto-detected (`AUTO_LANGUAGES` plus learned languages). `whisper_languages`: every code Whisper supports; codes not in `languages` are transcribed by Whisper, not translated |
| `GET` | `/` | The voice chat UI |

### `POST /api/transcribe`

Send `multipart/form-data`:

| Field | Required | Description |
|---|---|---|
| `audio` | yes | Audio file: webm, ogg, mp4/m4a, wav, mp3, flac, aiff, ... |
| `language` | no | Spoken language: `en`, `hi`, `mr`, `ta`, ... (see `/api/languages`). Leave it out to auto-detect |
| `output` | no | `english` (default): Indian languages are translated to English. `original`: return the transcript in the spoken language |
| `beam_size` | no | 1–10, default 5. Whisper only (English and other languages). Lower is faster |

curl (in Windows PowerShell, type `curl.exe`):

```bash
curl -F "audio=@meeting.mp3" http://127.0.0.1:8000/api/transcribe                   # auto-detect, English out
curl -F "audio=@call.m4a" -F "language=mr" -F "output=original" http://127.0.0.1:8000/api/transcribe
```

Python (`pip install requests` first):

```python
import requests

with open("meeting.mp3", "rb") as f:
    r = requests.post("http://127.0.0.1:8000/api/transcribe", files={"audio": f})
print(r.json()["text"])
```

JavaScript (browser; from a page on another address, also set `CORS_ORIGINS`):

```js
const form = new FormData();
form.append("audio", audioBlob, "recording.webm");
const res = await fetch("http://127.0.0.1:8000/api/transcribe", { method: "POST", body: form });
const { text } = await res.json();
```

Response:

```json
{
  "text": "Madagascar is by far the largest in terms of life and is a continent in itself.",
  "transcript": "जीवांचा विचार केल्यास मादागास्कर आतापर्यंत सर्वात मोठे आहे आणि एक स्वतःच खंड आहे",
  "language": "mr",
  "language_name": "Marathi",
  "language_probability": 1.0,
  "translated": true,
  "engine": "indic",
  "duration": 7.2,
  "processing_time": 0.24,
  "segments": [{"start": 0.0, "end": 7.2, "text": "Madagascar is by far ...", "transcript": "जीवांचा विचार ..."}]
}
```

| Field | Meaning |
|---|---|
| `text` | The final text: English when `translated` is true, otherwise the transcript |
| `transcript` | What was said, in the spoken language's own script |
| `language`, `language_name` | Spoken language code, and its name for English and the 22 Indian languages (for other languages the name is just the code) |
| `language_probability` | Detection confidence: 1.0 when you sent `language` or the clip was redone as English, 0.0 when no speech was found, `null` when not available (e.g. Odia) |
| `translated` | Whether the text was translated to English |
| `engine` | `whisper` (English and other languages) or `indic` (Indian languages) |
| `segments` | Timed pieces with `text` and `transcript`. Long Indian-language audio is split at pauses into pieces of up to 30 s |

For English, `text` and `transcript` are the same. If there's no speech, `text` is `""`,
`engine` is `null` and `language` is `null` (unless you sent one).

### Errors

Errors return `{"detail": "..."}` (for `422`, `detail` is a list of validation errors). The one exception is `Invalid host header`, which is plain text.

| Status | Meaning |
|---|---|
| `400` | Not a valid audio file, empty file or no samples, unsupported language, or `output` not `english`/`original`. Also `Invalid host header` when the Host isn't in `ALLOWED_HOSTS` |
| `403` | Browser request from another website (see `CORS_ORIGINS`) |
| `411` | Upload without a `Content-Length` header. Browsers, `curl -F` and Python `requests` always send it |
| `413` | Upload larger than `MAX_UPLOAD_MB`, or audio longer than `MAX_AUDIO_SECONDS` |
| `422` | Missing `audio` field or invalid `beam_size` |
| `503` | Models still loading, failed to load, or server busy. Retry shortly |

---

## Configuration

Set these environment variables before `python server.py`:

| Variable | Default | Description |
|---|---|---|
| `WHISPER_MODEL` | `large-v3-turbo` | Whisper model for English (see [Choosing a model](#choosing-a-model)) |
| `INDIC` | `1` | `0` turns off the Indian-language pipeline: Whisper only, ~2 GB of RAM, no torch needed |
| `INDIC_DECODING` | `rnnt` | IndicConformer decoder: `rnnt` (slightly more accurate) or `ctc` |
| `AUTO_LANGUAGES` | `en,hi,mr,bn,gu,pa,ta,te,kn,ml,or,as,ne` | Languages auto-detection starts from. Rarely needed: the server learns which languages you speak |
| `LANGUAGE_MEMORY` | `~/.cache/speech-to-text/language_memory.json` | Where the learned languages are saved. To reset: stop the server, delete the file, start it again |
| `INDIC_ASR_REPO` | *(ungated mirror)* | Set to `ai4bharat/indic-conformer-600m-multilingual` to use the official repo (accept its terms on Hugging Face and run `hf auth login` first). Its pinned revision is picked automatically |
| `INDIC_ASR_REVISION` | *(pinned per repo)* | Override the IndicConformer revision |
| `HOST` | `127.0.0.1` | `0.0.0.0` accepts connections from other machines |
| `PORT` | `8000` | Port |
| `MAX_UPLOAD_MB` | `25` | Max upload size |
| `MAX_AUDIO_SECONDS` | `1800` | Max audio length (30 min). The UI stops recording at 5 min |
| `MAX_CONCURRENT` | `2` | Transcriptions running at once. Extra requests wait up to 5 min |
| `ALLOWED_HOSTS` | *(none)* | Extra host names or IPs the server answers to (`*` = any). `127.0.0.1` and `localhost` always work |
| `CORS_ORIGINS` | *(none)* | Extra browser origins allowed to call the API, comma-separated (`*` = any website) |

Examples:

```bash
WHISPER_MODEL=small PORT=8001 python server.py        # macOS / Linux
```

```powershell
$env:WHISPER_MODEL="small"; $env:PORT="8001"; python server.py   # Windows PowerShell
```

### Calling the API from your own web app

If your front end runs on another address (for example `http://localhost:3000`), allow it:

```bash
CORS_ORIGINS=http://localhost:3000 python server.py
```

To reuse `index.html` there, set `API_BASE` at the top of its `<script>` to
`"http://127.0.0.1:8000"`.

### Opening `index.html` directly (double-click)

Start the server with `CORS_ORIGINS=null`. If the browser won't record from a `file://`
page, use http://127.0.0.1:8000 instead. That always works.

### Access from other devices

Start with `HOST=0.0.0.0` and add this machine's IP to `ALLOWED_HOSTS`, for example
`HOST=0.0.0.0 ALLOWED_HOSTS=192.168.1.20 python server.py`. Other devices can then call
the API at `http://192.168.1.20:8000`.

To **record** from a phone or another computer, the page must be served over HTTPS, for
example behind a reverse proxy such as Caddy or nginx:

- **Proxy keeps the original Host** (Caddy by default, or nginx with `proxy_set_header Host $host;`):
  add the public host name to `ALLOWED_HOSTS`, for example `ALLOWED_HOSTS=stt.example.com`.
  Without it, every request gets `400 Invalid host header`.
- **Proxy rewrites Host to the upstream** (nginx default): add the public origin to
  `CORS_ORIGINS`, for example `CORS_ORIGINS=https://stt.example.com`. Without it, POSTs
  get `403 Cross-origin request blocked`.

---

## How it learns your languages

There's nothing to set up. Every time the server confidently recognises an Indian language, or you
choose one (dropdown, or the API's `language` field), it notes it in a small file
(`~/.cache/speech-to-text/language_memory.json`, on Windows under your user folder).

- Once at least one language is learned, every other candidate language gets a small penalty;
  learned languages get none. Nothing changes until the first language is learned. All learned
  languages are treated equally, so Hindi and Marathi never compete through it.
- It only tips close calls, e.g. a short Hindi phrase that sounds like Nepali or Punjabi, or Hindi
  vs Marathi if you only speak one of them. (If you speak both, the words used decide.) Clearly
  different speech (say you start speaking Tamil) still wins, and English detection isn't affected.
- Old usage fades (each newly recorded use counts older ones at 97%), so it keeps adapting. A language you
  once picked manually (e.g. Konkani) also joins auto-detection.
- Only Indian languages are tracked. English detection doesn't need it and isn't influenced by it.
- Progress is visible at http://127.0.0.1:8000/api/health:
  - `language_memory`: how much each language has been heard (a dropdown pick counts double).
  - `learned_languages`: languages with enough use (about two confident uses, or one dropdown pick).
- The server learns only from confident detections (not from clips with no words or that turned out
  to be English), so an occasional wrong guess doesn't stick.
- To reset: stop the server, delete the file, then start it again (the running server keeps a copy
  in RAM and would write it back).

---

## Choosing a model

This applies to English (Whisper). The Indian-language models are fixed.

| Model | Accuracy | Speed on CPU | RAM (int8) |
|---|---|---|---|
| `large-v3-turbo` (default) | Excellent | Fast | ~1.5 GB |
| `large-v3` | Best | Slow | ~2.5 GB |
| `distil-large-v3` | Excellent (English only) | Fast | ~1.5 GB |
| `medium` | Very good | Medium | ~1 GB |
| `small` | Good | Very fast | ~0.5 GB |
| `base` / `tiny` | Basic | Fastest | < 0.3 GB |

Each model downloads the first time it's used and is cached after that.

---

## Platform notes

- **Fully supported:** Windows x64, Apple Silicon Macs, Linux x86_64/aarch64, on Python 3.10–3.13.
  Python 3.14 also works on these, except Intel Macs.
- **Intel Macs: English only.** PyTorch stopped at 2.2.2 for Intel Macs and SpeechBrain needs
  2.4+, so `requirements.txt` skips torch and SpeechBrain there. Run with `INDIC=0`.
  Use Python 3.10–3.13 on macOS 13+, or 3.10–3.12 on macOS 11–12 (onnxruntime has no
  Intel-Mac build for Python 3.14).
- **Windows on ARM:** install the **x64** build of Python (ctranslate2 has no ARM64
  Windows build). It runs under emulation.
- **Linux:** install CPU-only PyTorch before `requirements.txt` (see [Quick start](#quick-start)).

---

## Models and licenses

| Model | Used for | License |
|---|---|---|
| [Whisper large-v3-turbo](https://huggingface.co/mobiuslabsgmbh/faster-whisper-large-v3-turbo) (OpenAI, via faster-whisper) | English | MIT |
| [IndicConformer-600M-multilingual](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual) (AI4Bharat) | Indian-language speech recognition | MIT |
| [IndicTrans2 indic→en distilled 200M](https://huggingface.co/ai4bharat/indictrans2-indic-en-dist-200M) (AI4Bharat), [ONNX export](https://huggingface.co/hari31416/indictrans2-indic-en-dist-200M-ONNX) | Translation to English | MIT |
| [VoxLingua107 ECAPA](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa) (SpeechBrain) | Language detection | Apache-2.0 |
| `english_detector.json` (this project) | English vs Indian, on VoxLingua107 features | Trained on [EdAcc](https://huggingface.co/datasets/edinburghcstr/edacc) Indian English (CC-BY-SA 4.0), [FLEURS](https://huggingface.co/datasets/google/fleurs) (CC-BY 4.0) and synthetic clips. Attribute these datasets if you redistribute it |

All are free for commercial use. Model revisions are pinned in `indic.py`. By default the
IndicConformer weights come from an ungated mirror that is byte-identical to AI4Bharat's
repo (403 of 404 files identical by git object ID; only the README differs). Use
`INDIC_ASR_REPO` to switch to the official one.
