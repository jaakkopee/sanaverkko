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


def _set_current_sound(samples):
    global _current_samples, _playback_position

    if not _ensure_audio():
        return

    with _state_lock:
        _current_samples = np.clip(np.array(samples, dtype=np.float32), -1.0, 1.0)
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


def generate_melody(melody, amplitude, sample_rate, duration_per_note=0.1):
    _ensure_audio(sample_rate=sample_rate)
    if melody is None:
        return

    notes = list(melody)
    if not notes:
        return

    chunks = []
    for note in notes:
        freq = float(note)
        if freq <= 0:
            sample_count = max(1, int(_sample_rate * duration_per_note))
            chunks.append(np.zeros(sample_count, dtype=np.float32))
            continue
        t = _create_timebase(duration_per_note)
        chunks.append(_build_wave(np.sin(2.0 * math.pi * freq * t), amplitude))

    _set_current_sound(np.concatenate(chunks))


def set_amplitude(amplitude):
    _ = amplitude


def set_freq(freq):
    _ = freq


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
