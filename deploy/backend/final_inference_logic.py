import os
from pathlib import Path
import sys
import torch
import torchaudio
import numpy as np
from transformers import Wav2Vec2Model
import torch.nn as nn
from aasist_backend import AASIST_Backend
# =========================
# CONFIG
# =========================
from huggingface_hub import hf_hub_download

# Fix model paths (use huggingface_hub caching to avoid strict copies or timeouts on Render)
def _resolve_model_path(filename):
    repo_id = "ritam-05/voice-detector-models"
    try:
        # This will download it to HuggingFace's global cache dir (~/.cache/huggingface/hub) and return its path
        print(f"Resolving model {filename} via Hugging Face Hub (this may take a minute if not cached)...")
        path = hf_hub_download(repo_id=repo_id, filename=filename)
        return path
    except Exception as e:
        raise FileNotFoundError(f"Failed to fetch {filename} from HF {repo_id}: {e}")

# Replace string device with torch.device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use bare filenames here; resolve paths inside load_models()
STAGE1_FILENAME = "stage1_detector_epoch3.pt"
STAGE2_HUMAN_FILENAME = "stage2_aasist_epoch3.pt"
STAGE2_AI_FILENAME = "stage2_aasist_epoch3.pt"
#fixed filenames
# STAGE1_FILENAME = "stage1.pt"
# STAGE2_HUMAN_FILENAME = "stage2_human.pt"
# STAGE2_AI_FILENAME = "stage2_aasist.pt"

SAMPLE_RATE = 16000
MIN_SECONDS = 5
CHUNK_SECONDS = 4
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS

# Stage-1 thresholds (more conservative): make Stage-1 less likely to early-return HUMAN
S1_HUMAN_THRESHOLD = 0.30               # very confident human required to short-circuit
S1_HUMAN_CHECK_THRESHOLD = 0.40   # Above this, we start verifying with AASIST
S1_HUMAN_RECHECK_THRESHOLD = 0.10 # Below this, we only overrule if Red Flags are EXTREME
S1_AI_CHECK_THRESHOLD = 0.60            # if s1 > this, perform extra Stage-2 confirmation

# Stage-2 thresholds (conservative; favor HUMAN)
S2_HUMAN_NON_STUDIO = 0.45
S2_AI_NON_STUDIO = 0.75

S2_HUMAN_STUDIO = 0.25
S2_AI_STUDIO = 0.90

# recheck thresholds
RECHECK_AI_STRONG = 0.55
RECHECK_AI_WEAK = 0.50

# lean threshold
LEAN_SCORE_THRESHOLD = 0.60

# AASIST (Stage-2 AI Verifier) thresholds
AASIST_AI_THRESHOLD = 0.50        # Lower threshold to confirm AI more easily
AASIST_HUMAN_THRESHOLD = 0.30     # Stricter threshold to overrule Stage-1 AI to HUMAN
AASIST_INCONCLUSIVE_RANGE = 0.20  # Range between thresholds

# =========================
# MODELS
# =========================
class Stage1Detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        h = self.encoder(x).last_hidden_state.mean(dim=1)
        return self.classifier(h).squeeze(1)


class Stage2VerifierAI(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        
        # keep encoder frozen
        for p in self.encoder.parameters():
            p.requires_grad = False
        
        # AASIST Backend
        self.backend = AASIST_Backend(input_dim=768)
    
    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x).last_hidden_state  # (B,T,768)
        
        return self.backend(features)



# =========================
# LOAD MODELS
# =========================
def load_models():
    # Resolve model paths at load time (do not raise on import)
    try:
        STAGE1_MODEL_PATH = _resolve_model_path(STAGE1_FILENAME)
        STAGE2_HUMAN_PATH = _resolve_model_path(STAGE2_HUMAN_FILENAME)
        # Try to resolve AI model, but if learning is in progress it might not exist yet
        try:
            STAGE2_AI_PATH = _resolve_model_path(STAGE2_AI_FILENAME)
        except Exception:
            STAGE2_AI_PATH = None
            print(f"Warning: AI Verifier model {STAGE2_AI_FILENAME} not found. AI verification will be disabled.")
    except FileNotFoundError as e:
        raise FileNotFoundError(str(e))

    try:
        # create and move models to the chosen DEVICE
        s1 = Stage1Detector()
        s1.load_state_dict(torch.load(STAGE1_MODEL_PATH, map_location=DEVICE))
        s1 = s1.to(DEVICE)
        s1.eval()
    except Exception as e:
        raise RuntimeError(f"Failed to load stage1 model from {STAGE1_MODEL_PATH}: {e}")

    try:
    # Use AASIST-based verifier instead of simple MLP
        s2_human = Stage2VerifierAI()
        s2_human.load_state_dict(torch.load(STAGE2_HUMAN_PATH, map_location=DEVICE))
        s2_human = s2_human.to(DEVICE)
        s2_human.eval()
    except Exception as e:
     raise RuntimeError(f"Failed to load stage2 human model from {STAGE2_HUMAN_PATH}: {e}")

    s2_ai = None
    if STAGE2_AI_PATH:
        try:
            s2_ai = Stage2VerifierAI()
            s2_ai.load_state_dict(torch.load(STAGE2_AI_PATH, map_location=DEVICE))
            s2_ai = s2_ai.to(DEVICE)
            s2_ai.eval()
        except Exception as e:
            print(f"Failed to load stage2 ai model from {STAGE2_AI_PATH}: {e}")
            s2_ai = None

    return s1, s2_human, s2_ai


# =========================
# AUDIO UTILS
# =========================
def spectral_flatness(wav: torch.Tensor) -> float:
    """Estimate spectral flatness (0..1). Lower => tonal (studio)"""
    wav_cpu = wav.detach().cpu()
    # STFT -> shape (freq_bins, frames)
    S = torch.stft(wav_cpu, n_fft=1024, hop_length=512, return_complex=True)
    power = S.abs().pow(2)
    # geometric mean across freq bins per frame
    geo = torch.exp(torch.log(power + 1e-12).mean(dim=0))
    arith = power.mean(dim=0)
    flatness = (geo / (arith + 1e-12)).mean().item()
    return float(flatness)


def estimate_snr_db(wav: torch.Tensor) -> float:
    wav_cpu = wav.detach().cpu()
    rms = wav_cpu.pow(2).mean().sqrt().item()
    noise_floor = torch.quantile(wav_cpu.abs(), 0.05).item()
    return float(20 * np.log10((rms + 1e-9) / (noise_floor + 1e-9)))


def is_studio_like(wav: torch.Tensor) -> bool:
    """Heuristic: high SNR and low spectral flatness indicate studio-like audio."""
    try:
        # Broaden studio detection: many clean human recordings sit around flatness 0.02-0.05.
        return estimate_snr_db(wav) > 20 and spectral_flatness(wav) < 0.05
    except Exception:
        return False


def apply_small_aug(wav: torch.Tensor) -> torch.Tensor:
    # tiny noise + slight random gain
    noise = torch.randn_like(wav) * 1e-4
    gain = 1.0 + (torch.randn(1).item() * 0.01)
    return (wav * gain + noise).clamp(-1.0, 1.0)


def tta_prob(model, chunk: torch.Tensor, n: int = 3) -> float:
    probs = []
    # ensure model params exist
    param_dtype = None
    try:
        param_dtype = next(model.parameters()).dtype
    except StopIteration:
        param_dtype = torch.float32
    for _ in range(n):
        aug = apply_small_aug(chunk)
        with torch.inference_mode():
            # move input to model device and dtype to avoid input/weight type mismatch
            try:
                inp = aug.unsqueeze(0).to(DEVICE, dtype=param_dtype)
            except Exception:
                inp = aug.unsqueeze(0).to(DEVICE).float()
            logit = model(inp)
            probs.append(torch.sigmoid(logit).item())
    return float(np.mean(probs))


def compute_lean(s2_mean: float, s2_frac: float, studio: bool = False) -> str:
    """Deterministic binary lean decision.

    Score = 0.75 * s2_mean + 0.25 * s2_frac
    If studio == True, slightly penalize AI by subtracting 0.05.
    Return 'AI' if score >= 0.5 else 'HUMAN'.
    """
    score = 0.75 * float(s2_mean) + 0.25 * float(s2_frac)
    if studio:
        score = score - 0.05
    return "AI" if score >= 0.60 else "HUMAN"


# --- Degradation-aware verifier helpers ---

def is_degraded(wav: torch.Tensor) -> bool:
    """Return True if waveform appears degraded: low SNR, high spectral flatness, or clipping."""
    try:
        snr = estimate_snr_db(wav)
        flat = spectral_flatness(wav)
        clipped = float((wav.abs() > 0.99).float().mean().item())
        return (snr < 18.0) or (flat > 0.08) or (clipped > 0.01)
    except Exception:
        return False


def is_augmented(wav: torch.Tensor) -> bool:
    """Detect compressed/augmented audio: low dynamics or unusually low flatness.

    Heuristics:
    - low RMS variance across frames (heavy compression)
    - very low spectral_flatness (over-processed / synthetic)
    """
    try:
        # RMS dynamics: frame-level RMS
        frame_len = 1024
        hop = 512
        frames = wav.unfold(0, frame_len, hop)
        rms = frames.pow(2).mean(dim=1).sqrt()
        dyn_ratio = float((rms.std().item() / (rms.mean().item() + 1e-9)))
        flat = spectral_flatness(wav)
        # heuristics thresholds (tunable)
        return (dyn_ratio < 0.08) or (flat < 0.008)
    except Exception:
        return False


# =========================
# EXPLANATION / LAYMAN HELPERS
# =========================

def detect_breaths(wav: torch.Tensor, sr: int = SAMPLE_RATE):
    """Return (found:bool, count:int) of short low-centroid bursts likely to be breaths."""
    try:
        frame_len = 2048
        hop = 512
        frames = wav.unfold(0, frame_len, hop).contiguous()
        if frames.size(0) == 0:
            return False, 0
        rms = frames.pow(2).mean(dim=1).sqrt()
        # spectral centroid per frame
        spec = torch.fft.rfft(frames, n=frame_len)
        mags = spec.abs()
        freqs = torch.linspace(0, sr / 2, mags.size(1))
        centroid = (mags * freqs).sum(dim=1) / (mags.sum(dim=1) + 1e-9)
        med = float(rms.median().item())
        # breaths: low centroid (<1kHz), modest RMS (not silence), short bursts
        breath_mask = (centroid < 1000.0) & (rms > med * 0.15) & (rms < med * 1.3)
        # group consecutive True frames into bursts and count short bursts
        hop_dur = hop / float(sr)
        max_burst_len_frames = int(0.5 / hop_dur) + 1
        count = 0
        i = 0
        bm = breath_mask.cpu().numpy()
        while i < bm.size:
            if bm[i]:
                j = i + 1
                while j < bm.size and bm[j]:
                    j += 1
                if (j - i) <= max_burst_len_frames:
                    count += 1
                i = j
            else:
                i += 1
        return (count > 0), int(count)
    except Exception:
        return False, 0


def estimate_f0_stats(wav: torch.Tensor, sr: int = SAMPLE_RATE):
    """Estimate simple F0 per voiced frame using autocorrelation. Returns (mean_f0, std_f0, voiced_frames_count)."""
    try:
        frame_len = 2048
        hop = 512
        frames = wav.unfold(0, frame_len, hop).contiguous()
        if frames.size(0) == 0:
            return 0.0, 0.0, 0
        rms = frames.pow(2).mean(dim=1).sqrt()
        med = float(rms.median().item())
        voiced_mask = rms > (med * 0.25)
        f0s = []
        fr_np = frames.cpu().numpy()
        for i, v in enumerate(voiced_mask.cpu().numpy()):
            if not v:
                continue
            frame = fr_np[i]
            frame = frame - frame.mean()
            if np.allclose(frame, 0.0):
                continue
            corr = np.correlate(frame, frame, mode="full")[len(frame)-1:]
            # restrict lags to plausible pitch (50..400 Hz)
            min_lag = int(sr / 400)
            max_lag = int(sr / 50)
            if max_lag <= min_lag or max_lag >= len(corr):
                continue
            window = corr[min_lag:max_lag]
            if window.size == 0:
                continue
            lag = np.argmax(window) + min_lag
            if corr[lag] < 1e-6:
                continue
            f0 = float(sr / float(lag))
            f0s.append(f0)
        if len(f0s) == 0:
            return 0.0, 0.0, 0
        return float(np.mean(f0s)), float(np.std(f0s)), len(f0s)
    except Exception:
        return 0.0, 0.0, 0


def repetition_score(chunks: list):
    """Compute a simple repetition score (0..1) by MFCC-mean cosine similarity across chunks.
    Refined: Sunday Suspense narrators have very high similarity (0.99). 
    We raise the AI-threshold to 0.995 to avoid false positives."""
    try:
        if not chunks or len(chunks) < 2:
            return 0.0
        
        mfcc_tf = torchaudio.transforms.MFCC(sample_rate=SAMPLE_RATE, n_mfcc=13)
        vecs = []
        for c in chunks:
            with torch.no_grad():
                v = mfcc_tf(c.unsqueeze(0)).mean(dim=2).squeeze(0).cpu().numpy()
                vecs.append(v)
        
        arr = np.stack(vecs, axis=0)
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
        arrn = arr / norms
        sim = np.dot(arrn, arrn.T)
        
        upper = sim[np.triu_indices(sim.shape[0], k=1)]
        if upper.size == 0:
            return 0.0
            
        max_sim = float(np.max(upper))
        # Narrators are consistent (0.98-0.99). AI synth engines often hit 0.999+
        return max_sim
    except Exception:
        return 0.0


def vocoder_artifact_score(wav: torch.Tensor, sr: int = SAMPLE_RATE):
    """Estimate likelihood of vocoder/processing artifacts.
    Refined: High-end studios have low flatness. AI has high-frequency 'shimmer' (>0.08 flatness)."""
    try:
        flat = spectral_flatness(wav)
        # Studio humans (Sunday Suspense) are around 0.02. Roger AI is 0.11.
        if flat < 0.05:
            return 0.0
            
        # AI often has compressed or 'thin' high-freq spectrum
        frame_len = 2048
        spec = torch.fft.rfft(wav, n=frame_len)
        mags = spec.abs()
        freqs = torch.linspace(0, sr / 2, mags.size(0))
        total = mags.sum().item() + 1e-9
        hf_mask = freqs > 6000.0
        hf_ratio = float(mags[hf_mask].sum().item() / total) if hf_mask.sum() > 0 else 0.0
            
        return float(flat * (1.1 - hf_ratio))
    except Exception:
        return 0.0


def detect_transients(wav: torch.Tensor, sr: int = SAMPLE_RATE):
    """Detect short high-frequency transients (mouth clicks, lip noises).
    Returns (found:bool, count:int, score:float 0..1)"""
    try:
        frame_len = 512
        hop = 256
        frames = wav.unfold(0, frame_len, hop)
        if frames.size(0) == 0:
            return False, 0, 0.0
        # compute short-time spectral flux in HF band
        spec = torch.fft.rfft(frames, n=frame_len)
        mags = spec.abs()
        freqs = torch.linspace(0, sr / 2, mags.size(1))
        hf_mask = freqs > 3000.0
        if hf_mask.sum().item() == 0:
            return False, 0, 0.0
        hf = mags[:, hf_mask]
        hf_energy = hf.sum(dim=1)
        med = float(hf_energy.median().item()) + 1e-9
        spikes = (hf_energy > med * 4.0).cpu().numpy()
        count = int(spikes.sum())
        score = float(min(1.0, count / 10.0))
        return (count > 0), count, score
    except Exception:
        return False, 0, 0.0


def reverb_score(wav: torch.Tensor, sr: int = SAMPLE_RATE):
    """Simple heuristic for reverberation: ratio of tail energy to overall energy (0..1)."""
    try:
        frame_len = 1024
        hop = 512
        frames = wav.unfold(0, frame_len, hop)
        if frames.size(0) < 5:
            return 0.0
        rms = frames.pow(2).mean(dim=1).sqrt().cpu().numpy()
        n = len(rms)
        tail = rms[int(n * 0.6):]
        if tail.size == 0:
            return 0.0
        tail_ratio = float(tail.mean() / (rms.mean() + 1e-9))
        return min(1.0, tail_ratio * 2.0)  # scale into 0..1
    except Exception:
        return 0.0


def get_audio_factors(chunks, full_wav):
    """Calculate hand-crafted audio factors for human/AI analysis."""
    factors = []

    # breathing
    breath_found, breath_count = detect_breaths(full_wav)
    factors.append({
        "name": "breathing",
        "value": breath_found,
        "score": float(min(1.0, breath_count / 3.0)),
        "conclusion": "HUMAN" if breath_found else "AI"
    })

    # pitch (F0) stats
    f0_mean, f0_std, f0_count = estimate_f0_stats(full_wav)
    if f0_count == 0:
        f0_score = 0.0
        f0_concl = "Neutral"
    else:
        # low std -> AI-like
        f0_score = float(max(0.0, min(1.0, (30.0 - f0_std) / 30.0)))
        f0_concl = "AI" if f0_std < 20.0 else "HUMAN"
    factors.append({"name": "pitch_variance", "value": f0_std, "score": f0_score, "conclusion": f0_concl})

    # dynamics
    frame_len = 1024
    hop = 512
    frames = full_wav.unfold(0, frame_len, hop)
    rms = frames.pow(2).mean(dim=1).sqrt()
    dyn_ratio = float((rms.std().item() / (rms.mean().item() + 1e-9)))
    dyn_score = float(max(0.0, min(1.0, (0.15 - dyn_ratio) / 0.15)))
    dyn_concl = "AI" if dyn_ratio < 0.08 else "HUMAN"
    factors.append({"name": "dynamics", "value": dyn_ratio, "score": dyn_score, "conclusion": dyn_concl})

    # repetition
    rep_score = repetition_score(chunks)
    factors.append({"name": "repetition", "value": rep_score, "score": rep_score, "conclusion": "AI" if rep_score > 0.7 else "HUMAN"})

    # clipping
    clipped_frac = float((full_wav.abs() > 0.99).float().mean().item())
    factors.append({"name": "clipping", "value": clipped_frac, "score": float(min(1.0, clipped_frac * 50.0)), "conclusion": "Neutral"})

    # studio
    studio = is_studio_like(full_wav)
    factors.append({"name": "studio_like", "value": bool(studio), "score": 1.0 if studio else 0.0, "conclusion": "Neutral"})

    # vocoder artifacts
    voc_score = vocoder_artifact_score(full_wav)
    factors.append({"name": "vocoder_artifacts", "value": voc_score, "score": float(min(1.0, voc_score * 20.0)), "conclusion": "AI" if voc_score > 0.04 else "HUMAN"})

    # transients / micro-noises
    trans_found, trans_count, trans_score = detect_transients(full_wav)
    factors.append({"name": "micro_noises", "value": trans_count, "score": trans_score, "conclusion": "HUMAN" if trans_found else "AI"})

    # reverb
    rb = reverb_score(full_wav)
    factors.append({"name": "reverb", "value": rb, "score": rb, "conclusion": "HUMAN" if rb > 0.2 else "AI"})

    return factors, breath_count, clipped_frac


def explain_layman(label: str, path: str, stage1=None, stage2_human=None, 
                   preloaded_chunks=None, preloaded_wav=None, preloaded_factors=None):
    """Compose a structured layman-friendly explanation from observable audio factors.

    Returns a dict with keys: text, confidence (0..1), factors (list of dicts), metrics.
    The text will explicitly note contradictions when label disagrees with the majority of indicators.
    """
    try:
        # Avoid reloading audio if pre-loaded data is provided
        if preloaded_chunks is not None and preloaded_wav is not None:
            chunks, full_wav = preloaded_chunks, preloaded_wav
        else:
            res = load_chunks(path)
            if res is None or res[0] is None:
                return {"text": "Audio too short or could not be processed", "confidence": 0.0, "factors": [], "metrics": {}}
            chunks, full_wav = res

        # Avoid recalculating factors if pre-loaded
        if preloaded_factors is not None:
            factors = preloaded_factors
            # attempt to extract breath_count and clipped_frac from preloaded factors or recompute
            breath_count = next((f["value"] if f["name"]=="breathing" else 0 for f in factors if f["name"]=="breathing"), 0)
            clipped_frac = next((f["value"] if f["name"]=="clipping" else 0.0 for f in factors if f["name"]=="clipping"), 0.0)
            if isinstance(breath_count, bool): # extract raw count if possible
                 _, breath_count = detect_breaths(full_wav)
        else:
            factors, breath_count, clipped_frac = get_audio_factors(chunks, full_wav)
        
        studio = any(f["value"] for f in factors if f["name"] == "studio_like")

        # compute some metrics for transparency
        try:
            snr_db = estimate_snr_db(full_wav)
        except Exception:
            snr_db = None
        try:
            flat = spectral_flatness(full_wav)
        except Exception:
            flat = None
        # spectral centroid std for voice bandwidth characterization
        try:
            frame_len2 = 1024
            hop2 = 512
            fr = full_wav.unfold(0, frame_len2, hop2).contiguous()
            spec = torch.fft.rfft(fr, n=frame_len2)
            mags = spec.abs()
            freqs = torch.linspace(0, SAMPLE_RATE / 2, mags.size(1))
            centroid = ((mags * freqs).sum(dim=1) / (mags.sum(dim=1) + 1e-9)).cpu().numpy()
            centroid_std = float(np.std(centroid))
        except Exception:
            centroid_std = None

        # quick s1/s2 metrics ONLY if not pre-calculated
        s1_mean = None
        s2_mean = None
        s2_frac = None
        try:
            if stage1 is not None and len(chunks) > 0:
                s1_quick = np.array([tta_prob(stage1, c, n=1) for c in chunks])
                s1_mean = float(s1_quick.mean())
            if stage2_human is not None and len(chunks) > 0:
                s2_quick = np.array([tta_prob(stage2_human, c, n=1) for c in chunks])
                s2_mean = float(s2_quick.mean())
                s2_frac = float((s2_quick > 0.5).mean())
        except Exception:
            pass

        metrics = {
            "s1_mean": s1_mean,
            "s2_mean": s2_mean,
            "s2_frac": s2_frac,
            "snr_db": snr_db,
            "flatness": flat,
            "centroid_std": centroid_std,
            "breaths": int(breath_count),
            "clipped_frac": float(clipped_frac),
        }



        # repetition of significant factors: sort by absolute score
        sorted_factors = sorted(factors, key=lambda x: x.get("score", 0.0), reverse=True)

        # quick confidence: prefer stage2 if available
        conf = None
        try:
            if s2_mean is not None:
                conf = s2_mean if label == "AI" else (1.0 - s2_mean)
        except Exception:
            conf = None

        # heuristics aggregator if stage2 not available
        if conf is None:
            ai_score = 0.0
            human_score = 0.0
            for f in factors:
                if f["conclusion"] == "AI":
                    ai_score += f["score"]
                elif f["conclusion"] == "HUMAN":
                    human_score += f["score"]
            # normalize
            total = ai_score + human_score + 1e-6
            conf = float(ai_score / total) if label == "AI" else float(human_score / total)
            # gentle prior
            conf = 0.15 + 0.7 * conf
            conf = min(max(conf, 0.01), 0.99)

        # count indicators
        ai_ind = sum(1 for f in factors if f["conclusion"] == "AI" and f.get("score", 0.0) > 0.2)
        hum_ind = sum(1 for f in factors if f["conclusion"] == "HUMAN" and f.get("score", 0.0) > 0.2)

        # choose top 3 human-friendly phrases
        phrases = []
        for f in sorted_factors[:4]:
            if f["name"] == "breathing":
                phrases.append("breathing found" if f["value"] else "no breathing detected")
            elif f["name"] == "pitch_variance":
                phrases.append("pitch very even" if f["conclusion"] == "AI" else "natural pitch variation")
            elif f["name"] == "dynamics":
                phrases.append("monotone / low dynamics" if f["conclusion"] == "AI" else "good dynamics / expressive delivery")
            elif f["name"] == "repetition":
                if f["value"] > 0.7:
                    phrases.append("repeated/looped parts detected")
            elif f["name"] == "vocoder_artifacts":
                if f["score"] > 0.2:
                    phrases.append("synthetic-like artifacts / thin high end")
            elif f["name"] == "micro_noises":
                if f["value"] > 0:
                    phrases.append("small mouth noises or clicks present")
            elif f["name"] == "clipping":
                if f["value"] > 0.01:
                    phrases.append("clipping or distortion present")
            elif f["name"] == "studio_like":
                if f["value"]:
                    phrases.append("studio-like clean recording (may mask cues)")
            elif f["name"] == "reverb":
                if f["value"] > 0.25:
                    phrases.append("room/reverb present")

        # Construct final text with contradiction handling
        conf_perc = int(conf * 100)
        indicator_note = ""
        if label == "HUMAN" and ai_ind >= max(1, hum_ind):
            # contradiction: label human but AI-like cues present
            indicator_note = " Likely human according to model, but AI-like cues observed (" + ", ".join([p for p in phrases if 'no breathing' in p or 'monotone' in p or 'synthetic' in p or 'repeated' in p]) + ")."
        elif label == "AI" and hum_ind >= max(1, ai_ind):
            indicator_note = " Likely AI according to model, but human cues observed (" + ", ".join([p for p in phrases if 'breathing' in p or 'good dynamics' in p or 'small mouth' in p or 'room/reverb' in p]) + ")."

        if label == "AI":
            text = f"AI — {', '.join(phrases) or 'no obvious human cues detected'}. Confidence ~ {conf_perc}%" + indicator_note
        elif label == "HUMAN":
            text = f"HUMAN — {', '.join(phrases) or 'human cues detected (breathing, dynamics, etc.)'}. Confidence ~ {conf_perc}%" + indicator_note
        else:
            text = f"INCONCLUSIVE — {', '.join(phrases) or 'insufficient cues'}. Confidence ~ {conf_perc}%" + indicator_note

        return {"text": text, "confidence": float(conf), "factors": factors, "metrics": metrics}
    except Exception:
        return {"text": "Could not generate layman explanation", "confidence": 0.0, "factors": [], "metrics": {}}


def make_degraded_variants(chunk: torch.Tensor):
    """Generate simple degraded variants of a chunk for re-checking."""
    variants = []
    # slightly higher noise
    variants.append(chunk + torch.randn_like(chunk) * 1e-3)
    # low-rate down/up sampling
    try:
        down = torchaudio.functional.resample(chunk, SAMPLE_RATE, 8000)
        up = torchaudio.functional.resample(down, 8000, SAMPLE_RATE)
        variants.append(up)
    except Exception:
        pass
    # soft compression
    variants.append(torch.tanh(chunk * 0.8))
    # combined: lowrate + noise
    try: 
        lr = torchaudio.functional.resample(chunk, SAMPLE_RATE, 11025)
        lr_up = torchaudio.functional.resample(lr, 11025, SAMPLE_RATE)
        variants.append(lr_up + torch.randn_like(lr_up) * 5e-4)
    except Exception:
        pass
    return variants


def degradation_recheck(stage2_model, chunks):
    """Run Stage-2 on degraded variants and aggregate scores (mean and fraction>0.5)."""
    probs = []
    for c in chunks:
        variants = make_degraded_variants(c)
        # include original chunk as well to be conservative
        variants.append(c)
        for v in variants:
            with torch.inference_mode():
                logit = stage2_model(v.unsqueeze(0).to(DEVICE))
                probs.append(torch.sigmoid(logit).item())
    if len(probs) == 0:
        return 0.0, 0.0
    arr = np.array(probs)
    return float(arr.mean()), float((arr > 0.5).mean())


def _ensure_wav(path):
    """Convert ALL audio (mp4, m4a, aac, wav, etc.) to a temp WAV via ffmpeg.
    This guarantees perfectly standard 16kHz, mono, s16 PCM audio to fix any 
    browser-uploaded or weirdly-encoded .WAV issues."""
    import subprocess, tempfile
    ext = os.path.splitext(path)[1].lower()
    
    # We purposefully do not early exit for .wav anymore, to enforce 16kHz & mono!
    # if ext in ('.wav',):
    #     return path, False
    fd, tmp_wav = tempfile.mkstemp(suffix='.wav')
    os.close(fd)
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-i', path, '-ac', '1', '-ar', str(SAMPLE_RATE), '-sample_fmt', 's16', tmp_wav],
            check=True, capture_output=True
        )
        return tmp_wav, True  # converted, caller should clean up
    except Exception as e:
        if os.path.exists(tmp_wav):
            os.unlink(tmp_wav)
        raise RuntimeError(f"ffmpeg conversion failed for {ext}: {e}")


def load_chunks(path):
    wav_path, needs_cleanup = _ensure_wav(path)
    try:
        wav, sr = torchaudio.load(wav_path)
    finally:
        if needs_cleanup and os.path.exists(wav_path):
            os.unlink(wav_path)
    wav = wav.mean(dim=0)

    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)

    # normalize
    wav = wav / (wav.abs().max() + 1e-9)

    # simple trim of leading/trailing silence (energy threshold)
    max_amp = wav.abs().max()
    if max_amp < 1e-6:
        return None, None
    thresh = max(0.02 * max_amp, 1e-4)
    indices = torch.where(wav.abs() > thresh)[0]
    if indices.numel() == 0:
        return None, None
    wav = wav[indices[0]:indices[-1] + 1]

    if len(wav) < SAMPLE_RATE * MIN_SECONDS:
        return None, None  # too short

    chunks = []
    step = CHUNK_SAMPLES // 2  # 50% overlap
    for i in range(0, len(wav) - CHUNK_SAMPLES + 1, step):
        chunks.append(wav[i:i + CHUNK_SAMPLES])

    return chunks, wav


# =========================
# INFERENCE LOGIC
# =========================
@torch.inference_mode()
def predict(path, stage1=None, stage2_human=None, stage2_ai=None, tta_n=None):
    # Ensure models are present (at least s1 is required)
    if stage1 is None:
        raise ValueError("Stage1 model not loaded.")
    # stage2_human is deprecated but kept for backward compatibility
    # Only stage2_ai (AASIST) is used for Stage 2 verification

    # ensure models are on DEVICE
    try:
        stage1.to(DEVICE)
        if stage2_ai:
            stage2_ai.to(DEVICE)
    except Exception:
        pass

    res = load_chunks(path)
    if res is None or res[0] is None:
        return "INCONCLUSIVE", 0.0, "Audio too short (<5s)", (None, None, None)

    chunks, full_wav = res

    # Stage-1 with reduced TTA (less aggressive to avoid inflating scores)
    s1_scores = []
    n_s1 = 1
    # n_s2 = tta_n if tta_n is not None else 3
    # n_s2_strong = tta_n if tta_n is not None else 5

    for c in chunks:
        s1_scores.append(tta_prob(stage1, c, n=n_s1))

    s1_scores = np.array(s1_scores)
    s1_mean = float(s1_scores.mean())

    studio = is_studio_like(full_wav)

    # ---- STAGE 1 DECISION ----
    # Get hand-crafted factors for extra verification ("human verifier features")
    hc_factors, breath_count, clipped_frac = get_audio_factors(chunks, full_wav)
    hc_ai_score = sum(f['score'] for f in hc_factors if f['conclusion'] == 'AI')
    hc_hum_score = sum(f['score'] for f in hc_factors if f['conclusion'] == 'HUMAN')
    
    # Specific red flags from hand-crafted features
    repetition = next((f['score'] for f in hc_factors if f['name'] == 'repetition'), 0.0)
    vocoder = next((f['score'] for f in hc_factors if f['name'] == 'vocoder_artifacts'), 0.0)
    
    # DECISION PATHWAY:
    # ---------------------------
    # TIER A: Extreme Human Certainty (Sunday Suspense / Professional Studio)
    # If the neural net is extremely confident it's human, we trust it over the hand-crafted flags.
    if s1_mean < 0.05:
         # Narrow threshold: Roger AI hits 0.9997, Sunday Suspense hits 0.9990.
         # The boundary of 'Impossible Consistency' is ~0.9992.
         if repetition > 0.9992:
              return "AI", 0.95, f"AI detected via impossible repetition ({repetition:.4f}); s1={s1_mean:.3f}", (chunks, full_wav, hc_factors)
         return "HUMAN", s1_mean, "HUMAN confirmed; clean studio recording", (chunks, full_wav, hc_factors)

    # TIER B: Red Flag Override (Human-mimicking AI like 'Roger')
    # If the voice is suspicious (S1 is not clearly human) or has strong flags.
    # Studio-quality human voices can look "clean" like AI, so require stronger flags there.
    is_suspicious = (s1_mean > 0.40)
    human_cues_present = (breath_count >= 1) or (hc_hum_score >= hc_ai_score + 0.15)
    if studio:
        has_strong_flags = (repetition > 0.9992) or (vocoder > 0.30)
    else:
        has_strong_flags = (repetition > 0.995) or (vocoder > 0.15)
    
    if is_suspicious or has_strong_flags:
        if stage2_ai is not None:
            n_ai_v2 = tta_n if tta_n is not None else 3
            s2_ai_scores = np.array([tta_prob(stage2_ai, c, n=n_ai_v2) for c in chunks])
            s2_ai_mean = float(s2_ai_scores.mean())
            
            # If red flags are strong and Stage-2 supports AI, mark AI.
            # For studio-like audio, require stronger Stage-2 support to avoid false positives.
            flag_support_threshold = 0.20 if studio else 0.05
            if has_strong_flags and s2_ai_mean >= flag_support_threshold:
                 return "AI", 0.95, f"AI detected via mechanical fingerprints; s1={s1_mean:.3f}, s2={s2_ai_mean:.3f}", (chunks, full_wav, hc_factors)
                 
            # Tiered logic for Stage-1 vs Stage-2
            s1_very_confident = s1_mean > 0.90
            s1_confident = s1_mean > 0.82
            
            if s1_very_confident:
                studio_ai_confirm = 0.30 if studio else 0.10
                if s2_ai_mean >= studio_ai_confirm and not human_cues_present:
                    return "AI", max(s1_mean, s2_ai_mean), f"Stage-1 very confident AI confirmed; s1={s1_mean:.3f}, s2={s2_ai_mean:.3f}", (chunks, full_wav, hc_factors)
                else:
                    return "HUMAN", (1.0 - s2_ai_mean), f"AI signals overruled by extremely confident AASIST Human signal; s2={s2_ai_mean:.3f}", (chunks, full_wav, hc_factors)
             
            # TIER 2: Stage-1 confident AI (0.82-0.90)
            elif s1_confident:
                tier2_ai_threshold = 0.65 if studio else 0.45
                tier2_human_threshold = 0.35 if studio else 0.25
                if s2_ai_mean >= tier2_ai_threshold and not human_cues_present:
                    combined_score = (s1_mean * 0.65) + (s2_ai_mean * 0.35)
                    return "AI", combined_score, f"Stage-1 confident AI, AASIST confirms; s1={s1_mean:.3f}, s2={s2_ai_mean:.3f}", (chunks, full_wav, hc_factors)
                elif s2_ai_mean < tier2_human_threshold:
                    if hc_hum_score > hc_ai_score + 0.3:
                        return "HUMAN", (1.0 - s2_ai_mean), f"Stage-1 AI overruled; AASIST + features indicate HUMAN; s2={s2_ai_mean:.3f}", (chunks, full_wav, hc_factors)
                    else:
                        return "INCONCLUSIVE", s1_mean, f"Stage-1 AI vs AASIST HUMAN, features unclear; s1={s1_mean:.3f}, s2={s2_ai_mean:.3f}", (chunks, full_wav, hc_factors)
                else:
                    weighted = (s1_mean * 0.6) + (s2_ai_mean * 0.4)
                    weighted_ai_threshold = 0.72 if studio else 0.65
                    if weighted >= weighted_ai_threshold and not human_cues_present:
                        return "AI", weighted, f"Weighted decision favors AI; s1={s1_mean:.3f}, s2={s2_ai_mean:.3f}, weighted={weighted:.3f}", (chunks, full_wav, hc_factors)
                    else:
                        return "INCONCLUSIVE", weighted, f"Borderline case; s1={s1_mean:.3f}, s2={s2_ai_mean:.3f}, weighted={weighted:.3f}", (chunks, full_wav, hc_factors)
             
            # TIER 3: Stage-1 moderately indicates AI (0.75-0.82)
            else:
                tier3_ai_threshold = 0.65 if studio else AASIST_AI_THRESHOLD
                tier3_human_threshold = 0.40 if studio else AASIST_HUMAN_THRESHOLD
                if s2_ai_mean >= tier3_ai_threshold and not human_cues_present:
                    combined_score = (s1_mean * 0.5) + (s2_ai_mean * 0.5)
                    return "AI", combined_score, f"Stage-1 moderate AI, AASIST confirms; s1={s1_mean:.3f}, s2={s2_ai_mean:.3f}", (chunks, full_wav, hc_factors)
                elif s2_ai_mean < tier3_human_threshold:
                    return "HUMAN", (1.0 - s2_ai_mean), f"Stage-1 AI overruled by AASIST; s2={s2_ai_mean:.3f}", (chunks, full_wav, hc_factors)
                else:
                    return "INCONCLUSIVE", s2_ai_mean, f"Both models uncertain; s1={s1_mean:.3f}, s2={s2_ai_mean:.3f}", (chunks, full_wav, hc_factors)
        else:
            if s1_mean > 0.90 or has_strong_flags:
                return "AI", s1_mean, f"Confirmed AI (S1/Flags), AASIST unavailable; s1={s1_mean:.3f}", (chunks, full_wav, hc_factors)
            else:
                return "INCONCLUSIVE", s1_mean, "Stage-1 AI, but AASIST not available for verification", (chunks, full_wav, hc_factors)
    
    # Stage-1 ambiguous (0.40-0.75)
    if stage2_ai is None:
        return "INCONCLUSIVE", s1_mean, "Stage-1 ambiguous, AASIST not available", (chunks, full_wav, hc_factors)
    
    n_ai_final = tta_n if tta_n is not None else 3
    s2_ai_scores = np.array([tta_prob(stage2_ai, c, n=n_ai_final) for c in chunks])
    s2_ai_mean = float(s2_ai_scores.mean())

    amb_ai_threshold = 0.70 if studio else 0.55
    amb_human_threshold = 0.45 if studio else 0.35
    if s2_ai_mean >= amb_ai_threshold and not human_cues_present:
        return "AI", s2_ai_mean, f"Stage-1 ambiguous, AASIST confident AI; s1={s1_mean:.3f}, s2={s2_ai_mean:.3f}", (chunks, full_wav, hc_factors)
    elif s2_ai_mean < amb_human_threshold:
        return "HUMAN", (1.0 - s2_ai_mean), f"Stage-1 ambiguous, AASIST leans HUMAN; s1={s1_mean:.3f}, s2={s2_ai_mean:.3f}", (chunks, full_wav, hc_factors)
    else:
        weighted = (s1_mean * 0.4) + (s2_ai_mean * 0.6)
        weighted_ai_threshold = 0.62 if studio else 0.52
        weighted_human_threshold = 0.50 if studio else 0.45
        if weighted >= weighted_ai_threshold and not human_cues_present:
            return "AI", weighted, f"Weighted decision leans AI; s1={s1_mean:.3f}, s2={s2_ai_mean:.3f}, weighted={weighted:.3f}", (chunks, full_wav, hc_factors)
        elif weighted < weighted_human_threshold:
            return "HUMAN", (1.0 - weighted), f"Weighted decision leans HUMAN; s1={s1_mean:.3f}, s2={s2_ai_mean:.3f}, weighted={weighted:.3f}", (chunks, full_wav, hc_factors)
        else:
            return "INCONCLUSIVE", weighted, f"Truly ambiguous case; s1={s1_mean:.3f}, s2={s2_ai_mean:.3f}, weighted={weighted:.3f}", (chunks, full_wav, hc_factors)






# =========================
# CLI
# =========================
def analyze_inconclusive(folder: str, out_csv: str = "inconclusive_report.csv", stage1=None, stage2_human=None):
    """Analyze files under `folder` and report whether INCONCLUSIVE files lean toward AI/HUMAN.

    Output CSV columns: file, s1_mean, s2_mean, s2_frac_gt_0.5, lean
    """
    print(f"Analyzing files under: {folder}")

    # allow preloaded models for efficiency
    if stage1 is None or stage2_human is None:
        stage1, stage2_human, _ = load_models()

    rows = []
    for root, _, files in os.walk(folder):
        for fn in files:
            if not fn.lower().endswith((".wav", ".mp3", ".flac", ".m4a")):
                continue
            path = os.path.join(root, fn)
            res = load_chunks(path)
            if res is None or res[0] is None:
                print(f"SKIP {path}: too short or unreadable")
                continue
            chunks, wav = res

            s1_scores = np.array([tta_prob(stage1, c, n=1) for c in chunks])
            s2_scores = np.array([tta_prob(stage2_human, c, n=3) for c in chunks])

            s1_mean = float(s1_scores.mean())
            s2_mean = float(s2_scores.mean())
            s2_frac_gt_05 = float((s2_scores > 0.5).mean())

            studio = is_studio_like(wav)
            lean = compute_lean(s2_mean, s2_frac_gt_05, studio=studio)
            rows.append({
                "file": path,
                "s1_mean": s1_mean,
                "s2_mean": s2_mean,
                "s2_frac_gt_0.5": s2_frac_gt_05,
                "lean": lean,
            })

            print(f"{fn} -> s1={s1_mean:.3f} s2={s2_mean:.3f} lean={lean} studio={studio}")

    # write CSV
    import csv
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file", "s1_mean", "s2_mean", "s2_frac_gt_0.5", "lean"])
        writer.writeheader()
        writer.writerows(rows)

    from collections import Counter
    c = Counter(r["lean"] for r in rows)
    print("\nSUMMARY:")
    for k in ("AI", "HUMAN", "AMBIGUOUS"):
        print(f"{k}: {c.get(k, 0)}")
    print(f"Saved report to {out_csv}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:\n  python final_inference_logic.py <audio.wav>\n  python final_inference_logic.py --analyze <folder> [out_csv]")
        sys.exit(1)

    if sys.argv[1] == "--analyze":
        if len(sys.argv) < 3:
            print("Usage: python final_inference_logic.py --analyze <folder> [out_csv]")
            sys.exit(1)
        folder = sys.argv[2]
        out_csv = sys.argv[3] if len(sys.argv) > 3 else "inconclusive_report.csv"
        analyze_inconclusive(folder, out_csv=out_csv)
        sys.exit(0)

    else:
        path = sys.argv[1]
        # Load models once for CLI usage and pass them into predict()
        try:
            stage1, stage2_human, stage2_ai = load_models()
        except Exception as e:
            print("Failed to load models:", e)
            sys.exit(1)

        label, score, reason, preloaded = predict(path, stage1=stage1, stage2_human=stage2_human, stage2_ai=stage2_ai)
        chunks, full_wav, hc_factors = preloaded

        print("\nFINAL VERDICT :")
        print("Label      :", label)
        print("Confidence :", round(score, 3))
        print("Reason     :", reason)
        # layman explanation for end users
        layman = explain_layman(label, path, stage1, stage2_human, 
                               preloaded_chunks=chunks, preloaded_wav=full_wav, preloaded_factors=hc_factors)
        print("Explanation (layman):", layman)
