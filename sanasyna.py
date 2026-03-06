import math
import threading
import wave

import numpy as np

try:
    import sounddevice as sd
except Exception:
    sd = None


_sample_rate = 44100
_audio_available = sd is not None
_stream = None
_stream_rate = None
_current_samples = np.zeros(1, dtype=np.float32)
_current_loop = True
_playback_position = 0
_is_playing = False
_state_lock = threading.Lock()
_adsr_attack = 0.01
_adsr_decay = 0.04
_adsr_sustain = 0.85
_adsr_release = 0.03


def _ensure_stream():
    global _stream, _stream_rate

    if not _audio_available:
        return False

    if _stream is not None and _stream_rate == _sample_rate:
        return True

    if _stream is not None:
        _stream.stop()
        _stream.close()
        _stream = None

    _stream = sd.OutputStream(
        samplerate=_sample_rate,
        channels=1,
        dtype="float32",
        callback=_audio_callback,
    )
    _stream.start()
    _stream_rate = _sample_rate
    return True


def _audio_callback(outdata, frames, timing_info, status):
    global _playback_position, _is_playing

    _ = timing_info
    _ = status

    with _state_lock:
        if not _is_playing or _current_samples.size == 0:
            outdata.fill(0)
            return

        samples = _current_samples
        sample_count = samples.size
        if sample_count == 0:
            outdata.fill(0)
            _is_playing = False
            return

        if _current_loop:
            indices = (np.arange(frames) + _playback_position) % sample_count
            chunk = samples[indices]
            _playback_position = (_playback_position + frames) % sample_count
        else:
            end_position = _playback_position + frames
            if end_position <= sample_count:
                chunk = samples[_playback_position:end_position]
                _playback_position = end_position
                if _playback_position >= sample_count:
                    _is_playing = False
            else:
                chunk = np.zeros(frames, dtype=np.float32)
                remaining = max(0, sample_count - _playback_position)
                if remaining > 0:
                    chunk[:remaining] = samples[_playback_position:sample_count]
                _playback_position = sample_count
                _is_playing = False

    outdata[:, 0] = chunk


def _ensure_audio(sample_rate=None, buffer_size=1024):
    global _sample_rate, _audio_available

    _ = buffer_size

    if sample_rate is not None:
        _sample_rate = int(sample_rate)

    if not _audio_available:
        return False
    return _ensure_stream()


def _create_timebase(duration):
    sample_count = max(1, int(_sample_rate * float(duration)))
    return np.arange(sample_count, dtype=np.float32) / float(_sample_rate)


def _apply_adsr(samples):
    sample_count = samples.size
    if sample_count <= 1:
        return samples

    attack_count = max(0, int(_adsr_attack * _sample_rate))
    decay_count = max(0, int(_adsr_decay * _sample_rate))
    release_count = max(0, int(_adsr_release * _sample_rate))

    total_shape_count = attack_count + decay_count + release_count
    if total_shape_count >= sample_count and total_shape_count > 0:
        scale = (sample_count - 1) / float(total_shape_count)
        attack_count = int(attack_count * scale)
        decay_count = int(decay_count * scale)
        release_count = int(release_count * scale)

    sustain_count = max(0, sample_count - attack_count - decay_count - release_count)
    envelope = np.zeros(sample_count, dtype=np.float32)

    cursor = 0
    if attack_count > 0:
        envelope[cursor:cursor + attack_count] = np.linspace(0.0, 1.0, attack_count, endpoint=False, dtype=np.float32)
        cursor += attack_count

    if decay_count > 0:
        envelope[cursor:cursor + decay_count] = np.linspace(1.0, _adsr_sustain, decay_count, endpoint=False, dtype=np.float32)
        cursor += decay_count

    if sustain_count > 0:
        envelope[cursor:cursor + sustain_count] = _adsr_sustain
        cursor += sustain_count

    if cursor < sample_count:
        remaining = sample_count - cursor
        release_start = _adsr_sustain if cursor > 0 else 1.0
        envelope[cursor:] = np.linspace(release_start, 0.0, remaining, endpoint=True, dtype=np.float32)

    return samples * envelope


def _set_current_sound(samples):
    global _current_samples, _playback_position

    if not _ensure_audio():
        return

    shaped_samples = _apply_adsr(np.array(samples, dtype=np.float32))

    with _state_lock:
        _current_samples = np.clip(shaped_samples, -1.0, 1.0)
        _playback_position = 0


def init_sanasyna():
    _ensure_audio()
    return 0


def init_audio(sample_rate=44100):
    _ensure_audio(sample_rate=sample_rate)


def play(loop=True):
    global _current_loop, _playback_position, _is_playing

    if not _ensure_audio():
        return

    with _state_lock:
        if _current_samples.size == 0:
            return
        _current_loop = bool(loop)
        _playback_position = 0
        _is_playing = True


def stop():
    global _is_playing, _playback_position

    with _state_lock:
        _is_playing = False
        _playback_position = 0


def close():
    global _stream, _stream_rate

    stop()
    if _stream is not None:
        _stream.stop()
        _stream.close()
        _stream = None
        _stream_rate = None


def _build_wave(samples, amplitude):
    return float(amplitude) * samples


def _wave_from_freq(freq, duration, waveform):
    t = _create_timebase(duration)
    wave_name = str(waveform or "sine").lower()

    if wave_name == "triangle":
        phase = (float(freq) * t) % 1.0
        return 4.0 * np.abs(phase - 0.5) - 1.0
    if wave_name == "square":
        return np.where(np.sin(2.0 * math.pi * float(freq) * t) >= 0.0, 1.0, -1.0)
    if wave_name == "sawtooth":
        phase = (float(freq) * t) % 1.0
        return 2.0 * phase - 1.0
    if wave_name == "noise":
        sample_count = max(1, int(_sample_rate * float(duration)))
        return np.random.uniform(-1.0, 1.0, sample_count)
    return np.sin(2.0 * math.pi * float(freq) * t)


def generate_sine_wave(freq, amplitude, sample_rate, duration=0.25):
    _ensure_audio(sample_rate=sample_rate)
    t = _create_timebase(duration)
    samples = _build_wave(np.sin(2.0 * math.pi * float(freq) * t), amplitude)
    _set_current_sound(samples)


def generate_square_wave(freq, amplitude, sample_rate, duration=0.25):
    _ensure_audio(sample_rate=sample_rate)
    t = _create_timebase(duration)
    raw = np.where(np.sin(2.0 * math.pi * float(freq) * t) >= 0.0, 1.0, -1.0)
    samples = _build_wave(raw, amplitude)
    _set_current_sound(samples)


def generate_sawtooth_wave(freq, amplitude, sample_rate, duration=0.25):
    _ensure_audio(sample_rate=sample_rate)
    t = _create_timebase(duration)
    phase = (float(freq) * t) % 1.0
    raw = 2.0 * phase - 1.0
    samples = _build_wave(raw, amplitude)
    _set_current_sound(samples)


def generate_triangle_wave(freq, amplitude, sample_rate, duration=0.25):
    _ensure_audio(sample_rate=sample_rate)
    t = _create_timebase(duration)
    phase = (float(freq) * t) % 1.0
    raw = 4.0 * np.abs(phase - 0.5) - 1.0
    samples = _build_wave(raw, amplitude)
    _set_current_sound(samples)


def generate_noise_wave(freq, amplitude, sample_rate, duration=0.25):
    _ensure_audio(sample_rate=sample_rate)
    _ = freq
    sample_count = max(1, int(_sample_rate * float(duration)))
    raw = np.random.uniform(-1.0, 1.0, sample_count)
    samples = _build_wave(raw, amplitude)
    _set_current_sound(samples)


def generate_melody(melody, amplitude, sample_rate, duration_per_note=0.1, waveform="sine"):
    _ensure_audio(sample_rate=sample_rate)
    if melody is None:
        return

    notes = list(melody)
    if not notes:
        return

    chunks = []
    for note in notes:
        note_duration = float(duration_per_note)
        note_freq = note

        if isinstance(note, dict):
            note_freq = note.get("freq", note.get("frequency", 0.0))
            if "duration" in note:
                note_duration = float(note["duration"])
        elif isinstance(note, (tuple, list)) and len(note) > 0:
            note_freq = note[0]
            if len(note) > 1:
                note_duration = float(note[1])

        note_duration = max(0.01, note_duration)
        freq = float(note_freq)
        if freq <= 0:
            sample_count = max(1, int(_sample_rate * note_duration))
            chunks.append(np.zeros(sample_count, dtype=np.float32))
            continue
        raw_wave = _wave_from_freq(freq, note_duration, waveform)
        chunks.append(_build_wave(raw_wave, amplitude))

    _set_current_sound(np.concatenate(chunks))


def set_amplitude(amplitude):
    _ = amplitude


def set_freq(freq):
    _ = freq


def set_adsr(attack=0.01, decay=0.04, sustain=0.85, release=0.03):
    global _adsr_attack, _adsr_decay, _adsr_sustain, _adsr_release

    _adsr_attack = max(0.0, float(attack))
    _adsr_decay = max(0.0, float(decay))
    _adsr_sustain = min(1.0, max(0.0, float(sustain)))
    _adsr_release = max(0.0, float(release))


def set_sample_rate(sample_rate):
    _ensure_audio(sample_rate=sample_rate)


def set_buffer(samples, sample_rate):
    _ensure_audio(sample_rate=sample_rate)
    _set_current_sound(np.array(samples, dtype=np.float32))


def set_sound_buffer(buffer):
    global _current_samples, _playback_position
    _ensure_audio()
    with _state_lock:
        _current_samples = np.array(buffer, dtype=np.float32)
        _playback_position = 0


def set_buffer_from_samples(samples, sample_rate, channels):
    _ = channels
    set_buffer(samples, sample_rate)


def set_buffer_from_file(filename):
    if not _ensure_audio():
        return

    with wave.open(filename, "rb") as wav_file:
        channel_count = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frame_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        raw_audio = wav_file.readframes(frame_count)

    if sample_width != 2:
        return

    audio_int16 = np.frombuffer(raw_audio, dtype=np.int16)
    if channel_count > 1:
        audio_int16 = audio_int16.reshape(-1, channel_count).mean(axis=1).astype(np.int16)

    audio_float = audio_int16.astype(np.float32) / 32767.0
    set_buffer(audio_float, frame_rate)


def set_sound_buffer_from_file(filename):
    set_buffer_from_file(filename)
