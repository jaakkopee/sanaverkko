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
- Add words directly from UI
- Import text files as reference database (append/replace mode)
- Clear current sentence/network words
- Live result display in network view:
  - sentence,
  - gematria totals,
  - numerological reduction chain to digital root
- Real-time synthesized audio (sounddevice backend)
- Audio waveform modes:
  - Dynamic
  - Pure sine
  - Noise-heavy
  - Classic analog
- ADSR controls + graphical ADSR envelope display
- Separate output window showing `output.txt` updates live

## Repository layout

- `sanaVerkkoCore.py` – main app (UI + simulation + rendering)
- `sanasyna.py` – audio synthesis backend
- `requirements.txt` – Python dependencies
- `output.txt` – generated sentence/gematria output
- sample text files (`input.txt`, `kalevala.txt`, etc.) for experimentation

## Requirements

- Python 3.11+ (tested here with Python 3.14)
- macOS/Linux/Windows
- Audio device available for `sounddevice`

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
2. **Network window** (pygame): neurons, connections, sentence + gematria display
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

### Word/database controls
- **Add word(s)**: add one or many words (space-separated)
- **Clear sentence**: removes current network words
- **Import mode**:
  - Append database
  - Replace database
- **Import .txt**: load reference words from file

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

## Notes

- This is an experimental project; behavior is intentionally exploratory.
- Real-time mutation depends heavily on current parameter values and reference database content.
