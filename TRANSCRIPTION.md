# Speech Transcription & Drill-End Detection

The pipeline detects the **drill end** from the audio ("first room is clear")
and trims all metrics and artifacts to `[drill_start, drill_end]`. Drill
*start* comes from the tracker (first entry crossing); drill *end* comes from
speech transcription.

## ASR engine — NeMo Parakeet-TDT

Speech-to-text is **NVIDIA NeMo Parakeet** (`nvidia/parakeet-tdt-0.6b-v2`, an
`EncDecRNNTBPEModel`). It is fast, accurate on English command speech, emits
word- and segment-level timestamps directly (no separate alignment model), and
— being RNN-T — does not hallucinate text on silence, so **no external VAD is
required**. The checkpoint (~2.4 GB) downloads to the HuggingFace cache on
first run.

`nemo_toolkit[asr]` keeps `torch>=2.6` (the repo's conda-forge torch 2.8 is
fine) and `numpy>=2`; it pins `transformers~=4.57`. The env caps
`setuptools<81` so `pkg_resources` (used by `libs/Track`) stays available.

## Streaming, forward-only, stop-at-confirmed-end

Transcription is **stream-compatible**: it never seeks to end-of-file and never
loads the whole clip. As soon as the tracker confirms the first entry, a
background worker (`src/utils/drill_stream.py`) reads the audio in forward
chunks from an `AudioChunkSource` (`src/utils/audio_stream.py`), grows a buffer,
and re-transcribes it each step. It **stops ~one confirmation chunk past the
drill end** rather than transcribing the whole tail. If `stop_at_drill_end`
(default), the video frame loop also stops at the located end frame.

**Drill-end rule** (`DrillEndDetector` in `src/utils/drill_window.py`): fire on
the **first** transcript segment at/after the drill start that contains every
required word (`drill_window_required_words`, default `room,clear`), confirmed
by **two consecutive passes** (LocalAgreement-2) so a transient mis-hearing of
"clear" cannot end the drill early. The end is the **segment (sentence)
boundary** so the trimmed audio ends at a natural pause, not mid-word. The same
scan/select core also serves the offline `compute_drill_window`.

**Cut-robustness:** if the audio ends (or a live conversation is cut) before a
confirmed "room clear", the detector reports `end_uncertain` and the pipeline
runs to the last available frame — never crashes.

## Optional denoise

`enable_denoise` runs Facebook Denoiser (`dns48`) before ASR via
`DemucsStreamer` (`DenoiseSession` in `src/utils/denoise.py`) — a purpose-built
real-time streamer that carries internal state across chunks, so streamed
denoising is smooth (no chunk-boundary artifacts). Only the transcription
consumes denoised audio; saved videos keep the original track. Off by default.

Install note: the `denoiser` package is a **post-step**, not in
`environment.yml` — `pip install --no-deps denoiser==0.1.5`. Its metadata
over-pins `hydra-core<1.1` (used only by its training CLI), which conflicts
with NeMo's hydra 1.3.x; the inference path used here (`pretrained` +
`DemucsStreamer`) needs only torch and runs fine alongside NeMo (verified).
If the package is absent, `DenoiseSession` logs a warning and passes raw
audio through — the pipeline never breaks.

## Config keys

| Key | Default | Meaning |
|---|---|---|
| `enable_transcription` | true | Run ASR + drill-end detection |
| `drill_window_enabled` | true | Auto-detect drill start/end and trim to it |
| `stop_at_drill_end` | true | Stop the frame loop at the located end |
| `asr_backend` | `parakeet` | ASR engine |
| `transcription_device` | `cpu` | NeMo device: `cpu`, `cuda`, or `mps` (`mps` verified ~15× faster per pass on Apple Silicon; default stays `cpu`) |
| `transcription_preroll_sec` | 5.0 | Audio context before the drill start |
| `asr_stream_chunk_sec` | 8.0 | Forward chunk size per pass |
| `asr_stream_confirm_sec` | 3.0 | Shorter chunk to confirm a tentative end |
| `drill_window_required_words` | `room,clear` | Phrase that marks the drill end |
| `drill_window_min_align_score` | 0.4 | Lenient word-confidence gate |
| `drill_window_grace_tail_sec` | 0.5 | Padding after the matched segment |
| `enable_denoise` | false | FB Denoiser pre-pass |

## Artifacts

`{basename}_Transcription.json` — schema in `src/utils/transcription.py`:
`{schema_version, source_video_basename, language, model, aligned,
audio_window, denoise, segments:[{id, start, end, text, words:[{word, start,
end, score}]}]}`. `{basename}_DrillWindow.json` — the resolved window and the
matched segment. `{basename}_denoised.wav` when denoise is on.
