"""Indian-language pipeline: spoken language ID, speech recognition and translation to English.

- Language ID: English vs not is decided by a small classifier on VoxLingua107 embeddings
  (english_detector.json), with Whisper's language ID in server.py as fallback when unsure. The Indian language is picked by fusing SpeechBrain VoxLingua107 with
  IndicConformer's own per-language scores. looks_like_english() catches English speech that
  IndicConformer wrote out phonetically.
- Speech recognition: AI4Bharat IndicConformer-600M-multilingual (22 languages, ONNX).
- Translation: AI4Bharat IndicTrans2 indic->en distilled 200M (ONNX).

All models are MIT/Apache-2.0 licensed and run on CPU. Revisions are pinned.
"""

import json
import os
import re
import threading
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from faster_whisper.vad import VadOptions, get_speech_timestamps
from huggingface_hub import snapshot_download

SAMPLE_RATE = 16000

# IndicConformer: byte-identical ungated mirror of ai4bharat/indic-conformer-600m-multilingual
# (403/404 files identical by git object id; only README differs). Override to use the official
# gated repo after accepting its terms: INDIC_ASR_REPO=ai4bharat/indic-conformer-600m-multilingual
ASR_REPO = os.environ.get("INDIC_ASR_REPO", "sunilmahendrakar/indic-conformer-600m-multilingual")
ASR_PINS = {  # same files, different commit histories
    "sunilmahendrakar/indic-conformer-600m-multilingual": "ff6bbf95f1e30b772248f0b2ddb1dd1f0a20b422",
    "ai4bharat/indic-conformer-600m-multilingual": "e9b71b369c048e2c6b634d4c131061c34e441179",
}
ASR_REVISION = os.environ.get("INDIC_ASR_REVISION", ASR_PINS.get(ASR_REPO, "main"))
MT_REPO = "hari31416/indictrans2-indic-en-dist-200M-ONNX"
MT_REVISION = "73ebc1d69f476345f26eeb2e62592c8bbfac120e"
LID_REPO = "speechbrain/lang-id-voxlingua107-ecapa"
LID_REVISION = "0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9"

LANGUAGE_NAMES = {
    "as": "Assamese", "bn": "Bengali", "brx": "Bodo", "doi": "Dogri", "gu": "Gujarati",
    "hi": "Hindi", "kn": "Kannada", "kok": "Konkani", "ks": "Kashmiri", "mai": "Maithili",
    "ml": "Malayalam", "mni": "Manipuri", "mr": "Marathi", "ne": "Nepali", "or": "Odia",
    "pa": "Punjabi", "sa": "Sanskrit", "sat": "Santali", "sd": "Sindhi", "ta": "Tamil",
    "te": "Telugu", "ur": "Urdu",
}
# IndicTrans2 source tags, matching the script IndicConformer writes for each language.
MT_TAGS = {
    "as": "asm_Beng", "bn": "ben_Beng", "brx": "brx_Deva", "doi": "doi_Deva", "gu": "guj_Gujr",
    "hi": "hin_Deva", "kn": "kan_Knda", "kok": "gom_Deva", "ks": "kas_Arab", "mai": "mai_Deva",
    "ml": "mal_Mlym", "mni": "mni_Mtei", "mr": "mar_Deva", "ne": "npi_Deva", "or": "ory_Orya",
    "pa": "pan_Guru", "sa": "san_Deva", "sat": "sat_Olck", "sd": "snd_Deva", "ta": "tam_Taml",
    "te": "tel_Telu", "ur": "urd_Arab",
}
# Languages detected automatically: English plus major Indian languages that VoxLingua107 knows.
# The rest can still be chosen with an explicit language hint. Urdu speech is treated as Hindi.
DEFAULT_AUTO_LANGUAGES = "en,hi,mr,bn,gu,pa,ta,te,kn,ml,or,as,ne"
FUSION_WEIGHT = 0.15  # weight of VoxLingua log-probs when choosing among Indian languages
MAX_CHUNK_SECONDS = 29.0

# English detection, fast path: a logistic regression on VoxLingua's speaker-independent embedding
# (english_detector.json), trained on real Indian-accented English + 12 Indian languages. Held out:
# 100% of Indian English and 98-100% of Indian-language clips, and confident on ~95% of clips.
ENGLISH_SURE, NOT_ENGLISH_SURE = 0.95, 0.05
# When the fast detector is unsure, Whisper's language ID decides (reliable on Indian-accented
# English, 99.8% on held-out EdAcc, but costs one ~2 s encoder pass).
WHISPER_EN_SHARE = 0.95  # P(en) / (P(en) + P(Indian languages))
WHISPER_EN_MIN = 0.2  # absolute P(en), so "no idea" (top guess e.g. Indonesian) isn't read as English
CONTENT_EN_SIMILARITY = 0.8


# --- Detecting English that IndicConformer wrote out phonetically (e.g. "वॉट आर माई ऑर्डर्स") ---
CONSONANTS = dict(zip(
    "कखगघङचछजझञटठडढणतथदधनपफबभमयरलळवशषसह",
    ["k", "kh", "g", "gh", "n", "ch", "chh", "j", "jh", "n", "t", "th", "d", "dh", "n", "t", "th", "d", "dh", "n",
     "p", "ph", "b", "bh", "m", "y", "r", "l", "l", "v", "sh", "sh", "s", "h"]))
NUKTA = {"क": "k", "ख": "kh", "ग": "g", "ज": "z", "ड": "r", "ढ": "rh", "फ": "f", "य": "y"}
DEVANAGARI_LANGS = {"hi", "mr", "ne", "sa", "kok", "mai", "doi", "brx", "sd"}
# Malayalam chillu letters (final consonants) -> consonant + virama, so transliteration keeps them.
CHILLU = str.maketrans({"\u0d7a": "\u0d23\u0d4d", "\u0d7b": "\u0d28\u0d4d", "\u0d7c": "\u0d30\u0d4d",
                        "\u0d7d": "\u0d32\u0d4d", "\u0d7e": "\u0d33\u0d4d", "\u0d7f": "\u0d15\u0d4d"})
# Common Hindi and Marathi function words; phonetically written English contains none of them.
FUNCTION_WORDS = set("""
है हैं था थी थे हो हूँ हूं के की का को में से पर ने और या कर करो करें करना कीजिए दो दें दीजिए दिखाओ बताओ भेजो
क्या कितने कितना कितनी कहाँ कहां कब कौन कैसे क्यों मेरा मेरे मेरी मुझे हम आप यह ये वह वो इस उस नहीं भी तो ही
आज कल अभी वाला वाले लिए साथ बाद पहले कुछ सब
आहे आहेत होते होता होती आणि किंवा च्या चा ची चे ला ना मध्ये वर काय किती कुठे केव्हा कोण कसे कसा माझे माझा
माझी मला आम्ही तुम्ही हे ते नाही पण उद्या करा करायचे द्या दाखवा सांगा पाठवा साठी नंतर आधी
""".split())


def _consonant_skeleton_indic(text: str, lang: str) -> str:
    from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator

    if lang == "ml":
        text = text.translate(CHILLU)
    if lang not in DEVANAGARI_LANGS:
        text = UnicodeIndicTransliterator.transliterate(text, lang, "hi")
    text = unicodedata.normalize("NFD", text)  # split nukta letters (ज़ -> ज + nukta)
    out = []
    for i, ch in enumerate(text):
        if ch in CONSONANTS:
            nukta = i + 1 < len(text) and text[i + 1] == "\u093c"
            out.append(NUKTA.get(ch, CONSONANTS[ch]) if nukta else CONSONANTS[ch])
    return _consonant_skeleton_latin("".join(out))


def _consonant_skeleton_latin(text: str) -> str:
    t = text.lower()
    for a, b in [("sh", "s"), ("ch", "c"), ("kh", "k"), ("gh", "g"), ("th", "t"), ("dh", "d"), ("ph", "f"),
                 ("bh", "b"), ("ck", "k"), ("qu", "kv"), ("q", "k"), ("x", "ks"), ("w", "v"), ("z", "j"),
                 ("c", "k"), ("y", "")]:
        t = t.replace(a, b)
    return re.sub(r"[^bdfgjklmnprstv]", "", t)


def looks_like_english(transcript: str, translation: str, lang: str) -> bool:
    """True when an Indian-language transcript is really English written phonetically: it sounds
    the same as its own English translation and has no Hindi/Marathi function words."""
    words = transcript.split()
    if len(words) < 2 or any(w in FUNCTION_WORDS for w in words):
        return False
    similarity = SequenceMatcher(None, _consonant_skeleton_indic(transcript, lang),
                                 _consonant_skeleton_latin(translation)).ratio()
    return similarity >= CONTENT_EN_SIMILARITY


# --- Hindi vs Marathi from the words used (decisive on short phrases the acoustics find close) ---
HINDI_WORDS = set("""
है हैं था थी थे हूँ हूं के की का को में से और क्या कितने कितना कितनी कहाँ कहां कब कौन कैसे क्यों मेरा मेरे मेरी
मुझे हम यह ये वह वो इस उस इसे उसे नहीं भी लिए साथ बाद पहले कुछ सब करो करें करना कीजिए दो दें दीजिए दिखाओ बताओ
भेजो गया गई गए रहा रही रहे जाता जाती होता होती होते वाला वाले वाली जो जब तक अपने अपनी
""".split())
MARATHI_WORDS = set("""
आहे आहेत आहोत होते होतं आणि किंवा मध्ये काय किती कुठे केव्हा कोण कसे कसा कशी माझे माझा माझी मला आम्ही तुम्ही
नाही पण उद्या आता करा करायचे द्या दाखवा सांगा पाठवा साठी नंतर आधी झाले झाला झाली असे असा असलेल्या म्हणून जे
जेव्हा त्या त्याचे त्याची त्याचा त्यांनी हा ही
""".split())
MARATHI_ENDINGS = re.compile(r"(च्या|चा|ची|चे|ांना|ांनी|ाला|मध्ये|साठी|ातून|ाहून|ांचा|ांची|ांचे)$")
WORD_EVIDENCE_WEIGHT = 0.2


def hindi_marathi_words(hindi_text: str, marathi_text: str):
    """(Hindi word count in the Hindi reading, Marathi word/ending count in the Marathi reading)."""
    hi = sum(w in HINDI_WORDS for w in hindi_text.split())
    mr = sum(w in MARATHI_WORDS or (len(w) > 3 and bool(MARATHI_ENDINGS.search(w))) for w in marathi_text.split())
    return hi, mr


class LanguageMemory:
    """Remembers which Indian languages this user actually speaks, so auto-detection leans towards them.

    Every language used gets the same standing (Hindi and Marathi never compete through the memory);
    languages never used get a small penalty that clearly different speech still overrides. Usage
    fades (each new utterance multiplies old weights by DECAY). Saved to disk between runs.
    Simulated on FLEURS: 89% -> 92% for a Hindi/Marathi speaker; switching to e.g. Tamil still works."""

    DECAY, ACTIVE_AT, PENALTY, CONFIDENT_MARGIN = 0.97, 1.5, 0.3, 0.05

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        try:
            self.weights = {k: float(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}
        except (OSError, ValueError):
            self.weights = {}

    def snapshot(self) -> dict:
        with self.lock:
            return {k: round(v, 2) for k, v in sorted(self.weights.items(), key=lambda kv: -kv[1]) if v >= 0.05}

    def active(self) -> set:
        with self.lock:
            return {lang for lang, w in self.weights.items() if w >= self.ACTIVE_AT}

    def record(self, lang: str, weight: float = 1.0) -> None:
        with self.lock:
            for k in self.weights:
                self.weights[k] *= self.DECAY
            self.weights[lang] = self.weights.get(lang, 0.0) + weight
            data = {k: round(v, 4) for k, v in self.weights.items() if v >= 0.01}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError:
            pass  # memory is a convenience; never fail a request over it


def speech_chunks(samples: np.ndarray) -> list:
    """Split audio into speech chunks of at most MAX_CHUNK_SECONDS (IndicConformer uses full
    attention, so long audio must be chunked). Returns [(start_sample, end_sample), ...]."""
    spans = get_speech_timestamps(samples, VadOptions(max_speech_duration_s=MAX_CHUNK_SECONDS))
    chunks = []
    for s in spans:
        if chunks and s["end"] - chunks[-1][0] <= MAX_CHUNK_SECONDS * SAMPLE_RATE:
            chunks[-1] = (chunks[-1][0], s["end"])  # merge short neighbours into one chunk
        else:
            chunks.append((s["start"], s["end"]))
    return chunks


class LanguageID:
    """SpeechBrain VoxLingua107 ECAPA spoken language ID (107 languages, ~50 ms per clip)."""

    def __init__(self):
        from speechbrain.inference.classifiers import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy

        path = snapshot_download(LID_REPO, revision=LID_REVISION)
        # SpeechBrain wants its own folder; copy instead of symlinking (symlinks need
        # admin/developer mode on Windows).
        savedir = Path(path).parent.parent / "speechbrain" / LID_REVISION
        self.model = EncoderClassifier.from_hparams(
            source=path, savedir=str(savedir), local_strategy=LocalStrategy.COPY, run_opts={"device": "cpu"},
            overrides={"pretrained_path": Path(path).as_posix()},  # weights from the pinned snapshot, not Hub main
        )
        enc = self.model.hparams.label_encoder
        self.labels = [enc.ind2lab[i].split(":")[0] for i in range(len(enc.ind2lab))]

    def analyze(self, samples: np.ndarray):
        """Returns (language probabilities, 256-d embedding) from one pass."""
        with torch.inference_mode():
            emb = self.model.encode_batch(torch.from_numpy(samples).unsqueeze(0))
            logp = self.model.mods.classifier(emb).squeeze(1)[0]
        probs = dict(zip(self.labels, torch.softmax(logp, dim=-1).tolist()))
        probs["hi"] = probs.get("hi", 0.0) + probs.pop("ur", 0.0)
        return probs, emb[0, 0].numpy()


class IndicASR:
    """IndicConformer-600M-multilingual: shared encoder, CTC and RNNT decoders, per-language vocab."""

    BLANK_ID = 256  # index of the blank token within each language's 257-token vocabulary
    SOS = 5632
    RNNT_MAX_SYMBOLS = 10
    PRED_LAYERS, PRED_HIDDEN = 2, 640

    def __init__(self, decoding: str = "rnnt"):
        path = Path(snapshot_download(ASR_REPO, revision=ASR_REVISION)) / "assets"
        self.decoding = decoding
        self.preprocessor = torch.jit.load(str(path / "preprocessor.ts"), map_location="cpu")
        self.path = path
        self.sessions = {}
        self.session_lock = threading.Lock()
        names = ["encoder", "ctc_decoder"]
        if decoding == "rnnt":
            names += ["rnnt_decoder", "joint_enc", "joint_pred", "joint_pre_net"]
        for name in names:
            self._session(name)
        self.vocab = json.loads((path / "vocab.json").read_text(encoding="utf-8"))
        # Boolean masks over the shared 5633-token output; the last token is the shared blank.
        self.masks = {k: np.array(v, dtype=bool) for k, v in json.loads((path / "language_masks.json").read_text()).items()}
        self.blank_global = len(next(iter(self.masks.values()))) - 1
        self.token_masks = {}
        for lang, m in self.masks.items():
            tokens = m.copy()
            tokens[self.blank_global] = False
            self.token_masks[lang] = torch.from_numpy(tokens)

    def _session(self, name: str) -> ort.InferenceSession:
        # The 22 small per-language RNNT output layers load on first use.
        with self.session_lock:
            if name not in self.sessions:
                self.sessions[name] = ort.InferenceSession(str(self.path / f"{name}.onnx"), providers=["CPUExecutionProvider"])
            return self.sessions[name]

    def encode(self, samples: np.ndarray):
        wav = torch.from_numpy(samples).unsqueeze(0)
        with torch.inference_mode():
            feats, length = self.preprocessor(input_signal=wav, length=torch.tensor([wav.shape[-1]]))
        enc, _ = self.sessions["encoder"].run(
            ["outputs", "encoded_lengths"], {"audio_signal": feats.numpy(), "length": length.numpy()})
        ctc = self.sessions["ctc_decoder"].run(["logprobs"], {"encoder_output": enc})[0][0]
        return enc, ctc

    def language_scores(self, ctc: np.ndarray) -> dict:
        """Mean log-probability mass each language's vocabulary gets on non-blank frames."""
        lp = torch.from_numpy(ctc).log_softmax(-1)
        speech = lp.argmax(-1) != self.blank_global
        frames = lp[speech] if speech.any() else lp
        return {lang: torch.logsumexp(frames[:, m], dim=-1).mean().item() for lang, m in self.token_masks.items()}

    def decode(self, enc: np.ndarray, ctc: np.ndarray, lang: str) -> str:
        if self.decoding == "ctc":
            logprobs = torch.from_numpy(ctc[:, self.masks[lang]])
            ids = torch.unique_consecutive(logprobs.argmax(-1))
            tokens = [self.vocab[lang][i] for i in ids.tolist() if i != self.BLANK_ID]
        else:
            tokens = [self.vocab[lang][i] for i in self._rnnt_greedy(enc, lang)]
        return "".join(tokens).replace("\u2581", " ").strip()

    def _rnnt_greedy(self, enc: np.ndarray, lang: str) -> list:
        post_net = self._session(f"joint_post_net_{lang}")
        joint_enc = self.sessions["joint_enc"].run(["output"], {"input": enc.transpose(0, 2, 1)})[0]
        hyp = [self.SOS]
        state = (np.zeros((self.PRED_LAYERS, 1, self.PRED_HIDDEN), dtype=np.float32),
                 np.zeros((self.PRED_LAYERS, 1, self.PRED_HIDDEN), dtype=np.float32))
        for t in range(joint_enc.shape[1]):
            f = joint_enc[:, t:t + 1, :]
            for _ in range(self.RNNT_MAX_SYMBOLS):
                g, _, s0, s1 = self.sessions["rnnt_decoder"].run(
                    ["outputs", "prednet_lengths", "states", "162"],
                    {"targets": np.array([[hyp[-1]]], dtype=np.int32),
                     "target_length": np.array([1], dtype=np.int32),
                     "states.1": state[0], "onnx::Slice_3": state[1]})
                g = self.sessions["joint_pred"].run(["output"], {"input": g.transpose(0, 2, 1)})[0]
                joint = self.sessions["joint_pre_net"].run(["output"], {"input": f + g})[0]
                token = int(post_net.run(["output"], {"input": joint})[0].argmax())
                if token == self.BLANK_ID:
                    break
                hyp.append(token)
                state = (s0, s1)
        return hyp[1:]


class IndicTranslator:
    """IndicTrans2 indic->English (distilled 200M), ONNX, greedy decoding.

    Pre/post-processing mirrors IndicTransToolkit's IndicProcessor for Indic->English, using
    only pure-Python packages (IndicTransToolkit has no Windows wheel). IndicConformer output
    has no digits, URLs or e-mails, so IndicProcessor's placeholder step is not needed.
    """

    # IndicProcessor's FLORES-code -> indicnlp language code table (subset used here).
    ISO = {"asm_Beng": "as", "ben_Beng": "bn", "brx_Deva": "hi", "doi_Deva": "hi", "gom_Deva": "kK",
           "guj_Gujr": "gu", "hin_Deva": "hi", "kan_Knda": "kn", "kas_Arab": "ur", "mai_Deva": "hi",
           "mal_Mlym": "ml", "mni_Mtei": "hi", "mar_Deva": "mr", "npi_Deva": "ne", "ory_Orya": "or",
           "pan_Guru": "pa", "san_Deva": "hi", "sat_Olck": "or", "snd_Deva": "hi", "tam_Taml": "ta",
           "tel_Telu": "te", "urd_Arab": "ur"}
    PUNC = [
        (re.compile(r"\r"), ""), (re.compile(r"\(\s*"), "("), (re.compile(r"\s*\)"), ")"),
        (re.compile(r"\s:\s?"), ":"), (re.compile(r"\s;\s?"), ";"), (re.compile(r"[`´‘‚’]"), "'"),
        (re.compile(r"[„“”«»]"), '"'), (re.compile(r"[–—]"), "-"), (re.compile(r"\.\.\."), "..."),
        (re.compile(r" %"), "%"), (re.compile(r"nº "), "nº "), (re.compile(r" ºC"), " ºC"),
        (re.compile(r" [?!;]"), lambda m: m.group(0).strip()), (re.compile(r", "), ", "),
    ]
    MULTISPACE = re.compile(r"[ ]{2,}")
    MAX_SOURCE_TOKENS = 256
    AFTER_PUNC = [
        (re.compile(r"\) ([\.!:?;,])"), r")\1"), (re.compile(r"(\d) %"), r"\1%"),
        (re.compile(r"\"([,\.]+)"), r'\1"'), (re.compile(r"(\d) (\d)"), r"\1.\2"),
    ]

    def __init__(self):
        from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
        from indicnlp.tokenize import indic_tokenize
        from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator
        from sacremoses import MosesDetokenizer
        from tokenizers import Tokenizer

        path = Path(snapshot_download(MT_REPO, revision=MT_REVISION, allow_patterns=[
            "*.onnx", "*.onnx.data", "tokenizer_*.json", "generation_config.json"]))
        self.src_tok = Tokenizer.from_file(str(path / "tokenizer_src.json"))
        self.tgt_tok = Tokenizer.from_file(str(path / "tokenizer_tgt.json"))
        self.meta = json.loads((path / "tokenizer_meta.json").read_text(encoding="utf-8"))
        gen = json.loads((path / "generation_config.json").read_text(encoding="utf-8"))
        self.start_id = int(gen.get("decoder_start_token_id", 2))
        self.eos_id = int(gen.get("eos_token_id", 2))
        opts = ["CPUExecutionProvider"]
        self.enc = ort.InferenceSession(str(path / "encoder_model.onnx"), providers=opts)
        self.dec = ort.InferenceSession(str(path / "decoder_model.onnx"), providers=opts)
        self.dec_past = ort.InferenceSession(str(path / "decoder_with_past_model.onnx"), providers=opts)
        self.num_layers = (len(self.dec.get_outputs()) - 1) // 4

        self.normalizers = {}
        self.normalizer_factory = IndicNormalizerFactory()
        self.tokenize = indic_tokenize.trivial_tokenize
        self.xlit = UnicodeIndicTransliterator()
        self.detok = MosesDetokenizer(lang="en")
        self.lock = threading.Lock()

    def preprocess(self, text: str, src_tag: str) -> str:
        iso = self.ISO[src_tag]
        for pattern, repl in self.PUNC:
            text = pattern.sub(repl, text)
        text = self.MULTISPACE.sub(" ", text)
        for pattern, repl in self.AFTER_PUNC:
            text = pattern.sub(repl, text)
        text = text.strip()
        with self.lock:
            if iso not in self.normalizers:
                self.normalizers[iso] = self.normalizer_factory.get_normalizer(iso)
            normalizer = self.normalizers[iso]
        text = " ".join(self.tokenize(normalizer.normalize(text.strip()), iso))
        if src_tag.split("_")[1] not in ("Arab", "Aran", "Olck", "Mtei", "Latn"):
            text = self.xlit.transliterate(text, iso, "hi").replace(" ् ", "्")
        return f"{src_tag} eng_Latn {text.strip()}"

    def translate(self, text: str, lang: str, max_new_tokens: int = 256) -> str:
        if not text.strip():
            return ""
        ids = [i if i < self.meta["src_dict_size"] else self.meta["unk_id"]
               for i in self.src_tok.encode(self.preprocess(text, MT_TAGS[lang])).ids]
        if len(ids) > self.MAX_SOURCE_TOKENS:
            # The model accepts 256 source tokens: translate very long transcripts in halves.
            words = text.split()
            if len(words) > 1:
                mid = len(words) // 2
                return (self.translate(" ".join(words[:mid]), lang, max_new_tokens) + " "
                        + self.translate(" ".join(words[mid:]), lang, max_new_tokens)).strip()
            ids = ids[: self.MAX_SOURCE_TOKENS - 1] + ids[-1:]  # keep the end-of-sentence token
        input_ids = np.array([ids], dtype=np.int64)
        mask = np.ones_like(input_ids)
        hidden = self.enc.run(["last_hidden_state"], {"input_ids": input_ids, "attention_mask": mask})[0]

        out, past = [], None
        token = self.start_id
        for step in range(max_new_tokens):
            feed = {"input_ids": np.array([[token]], dtype=np.int64), "encoder_attention_mask": mask}
            if step == 0:
                result = self.dec.run(None, {**feed, "encoder_hidden_states": hidden})
            else:
                for i in range(self.num_layers):
                    for j, kind in enumerate(("decoder.key", "decoder.value", "encoder.key", "encoder.value")):
                        feed[f"past_key_values.{i}.{kind}"] = past[i * 4 + j]
                result = self.dec_past.run(None, feed)
            past = result[1:]
            token = int(result[0][0, -1].argmax())
            if token == self.eos_id:
                break
            out.append(token)
        ids = [i if i < self.meta["tgt_dict_size"] else self.meta["unk_id"] for i in out]
        raw = self.tgt_tok.decode(ids, skip_special_tokens=True)
        return self.detok.detokenize(raw.split(" "))


class IndicPipeline:
    def __init__(self, decoding: str = "rnnt", auto_languages: str = DEFAULT_AUTO_LANGUAGES):
        self.lid = LanguageID()
        head = json.loads((Path(__file__).resolve().parent / "english_detector.json").read_text(encoding="utf-8"))
        self.en_coef, self.en_intercept = np.array(head["coef"], dtype=np.float32), float(head["intercept"])
        self.asr = IndicASR(decoding)
        self.mt = IndicTranslator()
        auto = [c.strip() for c in auto_languages.split(",") if c.strip()]
        self.auto_english = "en" in auto
        self.auto_indic = [c for c in auto if c in LANGUAGE_NAMES and c != "ur"]
        memory_path = os.environ.get("LANGUAGE_MEMORY", str(Path.home() / ".cache" / "speech-to-text" / "language_memory.json"))
        self.memory = LanguageMemory(Path(memory_path))

    def candidates(self) -> list:
        """Auto-detect languages, plus any other Indian language this user has actually used."""
        return self.auto_indic + sorted(l for l in self.memory.active() if l in LANGUAGE_NAMES and l not in self.auto_indic and l != "ur")

    def quick_check(self, samples: np.ndarray, chunks: list):
        """~50 ms on up to 30 s of speech: VoxLingua107 language probabilities, and the probability
        that the speech is English (0 when English isn't an auto-detect language)."""
        speech = np.concatenate([samples[s:e] for s, e in chunks])[: 30 * SAMPLE_RATE]
        probs, emb = self.lid.analyze(speech)
        p_english = 1.0 / (1.0 + np.exp(-(float(emb @ self.en_coef) + self.en_intercept)))
        return probs, (p_english if self.auto_english else 0.0)

    def plausible(self, vox: dict) -> bool:
        """False when VoxLingua is sure it's none of the auto-detect languages (e.g. Chinese).
        Real English or Indian speech scored >= 8e-4 even on 1 s fragments in testing."""
        candidates = (["en"] if self.auto_english else []) + self.auto_indic
        return sum(vox.get(c, 0.0) for c in candidates) >= 1e-4

    def whisper_says_english(self, whisper_probs: dict) -> bool:
        if not self.auto_english:
            return False
        en = whisper_probs.get("en", 0.0)
        indic = sum(whisper_probs.get(c, 0.0) for c in self.auto_indic) + whisper_probs.get("ur", 0.0)
        return en >= WHISPER_EN_MIN and en / (en + indic) >= WHISPER_EN_SHARE

    def share(self, vox: dict, lang: str):
        """VoxLingua's probability for lang among the auto-detect Indian languages it knows."""
        known = [c for c in self.auto_indic if c in vox]
        total = sum(vox[c] for c in known)
        return vox[lang] / total if lang in vox and total > 0 else None

    def choose_indic(self, vox_probs: dict, encoded: list, candidates: list):
        """Pick the Indian language: IndicConformer's per-language scores (length-weighted over chunks)
        + VoxLingua + the user's language memory, then Hindi-vs-Marathi word evidence.
        Returns (language, confident, decoded) where decoded caches per-chunk transcripts."""
        weights = np.array([ctc.shape[0] for _, ctc in encoded], dtype=np.float64)
        scores = {}
        per_chunk = [self.asr.language_scores(ctc) for _, ctc in encoded]
        known = [vox_probs[c] for c in candidates if c in vox_probs]
        neutral = float(np.mean(known)) if known else 1.0  # for languages VoxLingua doesn't know (Odia)
        active = self.memory.active()
        for c in candidates:
            ic = np.average([s[c] if c != "hi" else max(s["hi"], s["ur"]) for s in per_chunk], weights=weights)
            vox = vox_probs.get(c, neutral)
            scores[c] = ic + FUSION_WEIGHT * np.log(vox + 1e-9)
            if active and c not in active:
                scores[c] -= LanguageMemory.PENALTY
        ranked = sorted(scores, key=scores.get, reverse=True)
        best = ranked[0]
        confident = len(ranked) < 2 or scores[ranked[0]] - scores[ranked[1]] >= LanguageMemory.CONFIDENT_MARGIN
        decoded = {}
        if best in ("hi", "mr") and "hi" in scores and "mr" in scores:
            decoded = {l: [self.asr.decode(e, ctc, l) for e, ctc in encoded] for l in ("hi", "mr")}
            hi_words, mr_words = hindi_marathi_words(" ".join(decoded["hi"]), " ".join(decoded["mr"]))
            hi_score = scores["hi"] + WORD_EVIDENCE_WEIGHT * hi_words
            mr_score = scores["mr"] + WORD_EVIDENCE_WEIGHT * mr_words
            best = "hi" if hi_score >= mr_score else "mr"
        return best, confident, decoded

    def transcribe(self, samples: np.ndarray, chunks: list, lang: str, vox_probs: dict,
                   translate: bool = True) -> dict:
        """Transcribe (and translate) speech chunks. lang="auto" picks the Indian language and also
        reports english_like=True when the speech was really English written out phonetically."""
        auto = lang == "auto"
        encoded = [self.asr.encode(samples[s:e]) for s, e in chunks]
        confident, decoded = False, {}
        if auto:
            lang, confident, decoded = self.choose_indic(vox_probs, encoded, self.candidates())
        segments = []
        for i, ((s, e), (enc, ctc)) in enumerate(zip(chunks, encoded)):
            text = decoded[lang][i] if lang in decoded else self.asr.decode(enc, ctc, lang)
            if not text:
                continue
            segments.append({
                "start": round(s / SAMPLE_RATE, 2),
                "end": round(e / SAMPLE_RATE, 2),
                "transcript": text,
                "text": self.mt.translate(text, lang) if translate else text,
            })
        english_like = False
        if auto and segments:
            transcript = " ".join(s["transcript"] for s in segments)
            translation = " ".join(s["text"] for s in segments) if translate else self.mt.translate(transcript, lang)
            english_like = looks_like_english(transcript, translation, lang)
        if segments and not english_like and (confident or not auto):
            # Learn from confident detections and from languages the user picked explicitly.
            self.memory.record(lang, 1.0 if auto else 2.0)
        return {"language": lang, "segments": segments, "english_like": english_like}
