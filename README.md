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
- ADSR controls + graphical ADSR envelope display
- Hybrid deterministic/random word selection (tunable exploration)
- Jump search controls to escape local minima:
  - Jump probability
  - Jump radius
  - Fluid root (on/off)
- Separate output window showing `output.txt` updates live

## Repository layout

- `sanaVerkkoCore.py` – main app (UI + simulation + rendering)
- `sanasyna.py` – audio synthesis backend
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

## Preset system

Presets serialize parameter values from the control window to JSON. Loading a preset updates the UI controls and applies the values immediately.

Bundled presets in `presets/`:
- `01_very_ordered.json` – highly stable, low exploration, no jumps
- `02_ordered_balanced.json` – mostly ordered with light variation
- `03_balanced_dynamic.json` – balanced default-style exploratory behavior
- `04_exploratory.json` – wider search and faster mutation
- `05_chaotic.json` – aggressive exploration/jumps and highly dynamic behavior

### Audio controls
- **Audio waveform mode**: Dynamic / Pure sine / Noise-heavy / Classic analog
- **ADSR envelope**:
  - Attack
  - Decay
  - Sustain
  - Release
  - plus graphical envelope preview

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

This prevents getting stuck in one fixed sentence while keeping transformations mostly logical.

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

## Notes

- This is an experimental project; behavior is intentionally exploratory.
- Real-time mutation depends heavily on current parameter values and reference database content.
