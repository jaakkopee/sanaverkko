# SanaVerkko

![SanaVerkko screenshot](sanaverkko%20screenshot%201.png)

SanaVerkko is an experimental **word network + gematria + audio** playground.

It combines:
- a pygame network visualization window,
- a wxPython control window,
- and a live output viewer window for `output.txt`.

You can add words, import a text database, mutate words by gematria relations, and hear the network state as synthesized audio.

## Features

- Gematria mapping for letters `a-z`, `å`, `ä`, `ö`
- Dynamic word-neuron network (connections weighted by gematria distance)
- Adjustable model parameters from UI
- Optional POS-aware word matching (toggle in UI)
- Optional long-term memory (LTM) guidance with external `.svltm` model
- Add words directly from UI
- Import text files as reference database (append/replace mode)
- JSON preset system (save/load from native file picker)
- Clear current sentence/network words
- Live result display in network view:
  - sentence,
  - gematria totals,
  - numerological reduction chain to digital root
- Real-time synthesized audio (sounddevice backend)
- Sentence melody synthesis from gematria patterns:
  - each word is converted to a per-letter gematria note pattern,
  - pattern is reflected on time axis,
  - all word patterns are concatenated and played as one looping melody
- Audio waveform modes:
  - Dynamic
  - Pure sine
  - Noise-heavy
  - Classic analog
- Frequency mapping modes for generated frequencies:
  - Original notes
  - Pythagorean pentatonic
  - Pythagorean 8 note
  - Equal tempered modal mappings (Ionian, Dorian, Frygian, Lydian, Mixolydian, Aeolian, Locrian)
  - Equal tempered 12 / 24 / 36 / 48 note mappings
- Polyphony controls:
  - Polyphony voices (1-4)
  - Voice spread
- Melody speed coefficient
- Minimum note duration with automatic duration scaling
- ADSR controls + graphical ADSR envelope display
- Hybrid deterministic/random word selection (tunable exploration)
- Jump search controls to escape local minima:
  - Jump probability
  - Jump radius
  - Fluid root (on/off)
- True connection-weight learning in backpropagation (learning rate / error / target driven)
- Separate output window showing `output.txt` updates live

## Repository layout

- `sanaVerkkoCore.py` – main app (UI + simulation + rendering)
- `sanasyna.py` – audio synthesis backend
- `sv_ltm.py` – long-term memory model training/loading CLI (`.svltm`)
- `requirements.txt` – Python dependencies
- `output.txt` – generated sentence/gematria output
- `presets/` – example JSON presets from ordered to chaotic
- sample text files (`input.txt`, `kalevala.txt`, etc.) for experimentation

## Requirements

- Python 3.11+ (tested here with Python 3.14)
- macOS/Linux/Windows
- Audio device available for `sounddevice`

Python dependencies used by this repo:
- `numpy`
- `pygame`
- `wxPython`
- `sounddevice`
- `nltk`

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Basic run (starts empty network by default):

```bash
python sanaVerkkoCore.py
```

Optional startup files:

```bash
python sanaVerkkoCore.py <input_sentence_file.txt> <reference_database_file.txt>
```

If only one file is provided, it is used as input words. Reference database can then be imported from the UI.

## Windows

When app starts, you should see:

1. **Control window** (wxPython): parameters, add/import, audio controls
2. **Network window** (pygame): neurons, connections, sentence + gematria display + POS tags
3. **Output window** (wxPython): live contents of `output.txt`

The control window is adaptive and scrollable.

## Control window quick guide

### Model controls
- Learning rate
- Error
- Target
- Activation increase
- Activation limit
- Sigmoid scale
- Word change threshold
- Zoom
- Process interval (seconds)
- Selection exploration (0-1)
- Selection top-k
- Jump probability (0-1)
- Jump radius (gematria)
- Use POS matching (checkbox)
- Use long term memory (checkbox)
- Fluid root (checkbox)

### Word/database controls
- **Add word(s)**: add one or many words (space-separated)
- **Clear sentence**: removes current network words
- **Import mode**:
  - Append database
  - Replace database
- **Import .txt**: load reference words from file

### Preset controls
- **Save preset**: save current parameters to a `.json` file
- **Load preset**: load parameters from a `.json` file
- Uses native file dialogs (same style as database import)

### LTM controls
- **Use long term memory**: enables LTM-assisted candidate ranking
- **Load long term memory**: loads an external `.svltm` model file
- LTM status line shows disabled/enabled/model-loaded state

### Audio controls
- **Audio waveform mode**: Dynamic / Pure sine / Noise-heavy / Classic analog
- **Frequency mapping**: choose mapping system for generated frequencies
- **Polyphony voices**: 1-4 voices
- **Voice spread**: detune/spread amount between voices
- **Melody speed coeff**: scales note durations
- **Minimum note duration (s)**: floor for shortest note; melody is scaled to respect this floor
- **ADSR envelope**:
  - Attack
  - Decay
  - Sustain
  - Release
  - plus graphical envelope preview

## Preset system

Presets serialize parameter values from the control window to JSON. Loading a preset updates the UI controls and applies the values immediately.

Bundled presets in `presets/`:
- `01_very_ordered.json` – highly stable, low exploration, no jumps
- `02_ordered_balanced.json` – mostly ordered with light variation
- `03_balanced_dynamic.json` – balanced default-style exploratory behavior
- `04_exploratory.json` – wider search and faster mutation
- `05_chaotic.json` – aggressive exploration/jumps and highly dynamic behavior
- `06_crystal_ordered_polyphony.json` – ordered behavior with richer polyphony
- `07_tight_balanced_triad.json` – balanced triad-like voice behavior
- `08_dynamic_glide_quartet.json` – dynamic four-voice exploratory profile
- `09_noisy_fractal_swarm.json` – high-motion, noisy exploratory profile
- `10_resonant_pulse_duo.json` – two-voice resonant pulse profile

Preset scale is intended to move from conservative/consistent behavior to highly exploratory/chaotic behavior.

## Output format (`output.txt`)

When sentence changes, app writes:
- generated sentence,
- per-word gematria sum expression,
- numerological reduction chain.

## Selection logic

- Word replacement uses a fast indexed lookup by:
  - exact gematria,
  - numerological reduction,
  - digital root,
  - and optionally POS tag.
- Candidate selection is hybrid:
  - deterministic best match by default,
  - with configurable exploration over top-ranked candidates.
- Jump mode can broaden candidate search:
  - `jump_probability` controls how often jump logic is used,
  - `jump_radius` controls gematria-distance search radius,
  - `fluid_root` controls whether jumps may cross digital roots.
- If LTM is enabled and model is loaded, candidate ranking is additionally guided by next-word probabilities from recent context.

This prevents getting stuck in one fixed sentence while keeping transformations mostly logical.

## Learning behavior

- `Neuron.backpropagate` updates connection weights directly.
- Weight updates are driven by:
  - `learning_rate`
  - `error`
  - `target`
  - local activation and connected neuron activation
- Weight values are clamped to a safe range to avoid runaway growth.

## Long-term memory model (`sv_ltm.py`)

- LTM is optional and loaded as an external model file (`.svltm`).
- The included implementation uses a Char-CNN + MLP with word-softmax next-word prediction.
- Core app integration:
  - load model from control window,
  - toggle LTM on/off,
  - re-rank replacement candidates by predicted probability.

For CLI training and parameter reference, see [LTM_USAGE.md](LTM_USAGE.md).

## Gematria, numerology, and POS background

Gematria maps letters to numbers and uses the resulting totals as symbolic features. Numerological reduction then repeatedly sums digits until a single-digit digital root is reached. In this project, those values are used as transformation signals and indexing keys for word replacement, not as scientific measurements of language truth.

Historically, gematria- and numerology-like methods appear in multiple ancient and medieval traditions and were discussed by scholars, philosophers, and mystics of their time. In modern mainstream science, these systems are generally treated as non-empirical or pseudoscientific. POS matching is a separate linguistic constraint intended to keep substitutions in roughly the same grammatical role.

## Parameter commit behavior

- Numeric text controls are **explicit commit** controls.
- New values are applied when you press **Enter** in that field.
- Invalid text is reverted to the last valid committed value.

## Troubleshooting

### No sound
- Ensure your output device works in system settings.
- Check dependency install: `pip install -r requirements.txt`
- On macOS, microphone/audio permissions may affect low-level audio access in some environments.

### UI controls not visible
- Use scrolling in control window.
- Resize the control window larger.

### Invalid words are rejected
Only words composed of letters in the gematria table are accepted (`a-z`, `å`, `ä`, `ö`).

### POS behavior
- If `nltk` tagger data is available, POS matching uses it.
- If not, app falls back to an internal heuristic POS tagger.

### Gematria weight crash (`Neuron` has no `gematria`)
- The connection-weight update path stores gematria on neuron objects and includes a safe fallback.
- If you still see this error in an old run, restart the app to ensure the latest code is loaded.

## Notes

- This is an experimental project; behavior is intentionally exploratory.
- Real-time mutation depends heavily on current parameter values and reference database content.
