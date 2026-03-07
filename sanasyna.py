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
_stream_blocksize = 2048
_stream_latency = "high"
_current_samples = np.zeros(1, dtype=np.float32)
_pending_samples = None
_current_loop = True
_playback_position = 0
_is_playing = False
_state_lock = threading.Lock()
_crossfade_seconds = 0.03
_callback_status_count = 0
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
        blocksize=int(_stream_blocksize),
        latency=_stream_latency,
        callback=_audio_callback,
    )
    _stream.start()
    _stream_rate = _sample_rate
    return True


def _audio_callback(outdata, frames, timing_info, status):
    global _playback_position, _is_playing, _pending_samples, _current_samples, _callback_status_count

    _ = timing_info
    if status:
        _callback_status_count += 1
        if _callback_status_count <= 8 or (_callback_status_count % 25 == 0):
            print(f"sanasyna callback status: {status}")

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

        if _pending_samples is not None and _pending_samples.size > 0:
            next_samples = _pending_samples
            _pending_samples = None

            crossfade_samples = max(16, int(_crossfade_seconds * float(_sample_rate)))
            crossfade_samples = min(crossfade_samples, frames, chunk.size, next_samples.size)

            if crossfade_samples > 0:
                fade_out = np.linspace(1.0, 0.0, crossfade_samples, endpoint=False, dtype=np.float32)
                fade_in = np.linspace(0.0, 1.0, crossfade_samples, endpoint=False, dtype=np.float32)
                mixed_chunk = np.array(chunk, copy=True)
                mixed_chunk[:crossfade_samples] = (
                    chunk[:crossfade_samples] * fade_out
                    + next_samples[:crossfade_samples] * fade_in
                )
                chunk = mixed_chunk
                _current_samples = next_samples
                _playback_position = crossfade_samples % next_samples.size
            else:
                _current_samples = next_samples
                _playback_position = 0
            _is_playing = True

    outdata[:, 0] = chunk


def _ensure_audio(sample_rate=None, buffer_size=1024):
    global _sample_rate, _audio_available, _stream_blocksize

    if sample_rate is not None:
        _sample_rate = int(sample_rate)
    if buffer_size is not None:
        _stream_blocksize = max(128, int(buffer_size))

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
    global _current_samples, _playback_position, _pending_samples

    if not _ensure_audio():
        return

    shaped_samples = _apply_adsr(np.array(samples, dtype=np.float32))

    with _state_lock:
        clipped = np.clip(shaped_samples, -1.0, 1.0)
        if _is_playing and _current_samples.size > 0:
            _pending_samples = clipped
        else:
            _current_samples = clipped
            _playback_position = 0
            _pending_samples = None


def set_transition_crossfade(seconds=0.03):
    global _crossfade_seconds
    _crossfade_seconds = min(0.25, max(0.0, float(seconds)))


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
        was_playing = _is_playing
        _current_loop = bool(loop)
        if not was_playing:
            _playback_position = 0
        _is_playing = True


def stop():
    global _is_playing, _playback_position, _pending_samples

    with _state_lock:
        _is_playing = False
        _playback_position = 0
        _pending_samples = None


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


def _parse_melody_notes(melody, default_duration, duration_coeff=1.0):
    parsed_notes = []
    coeff = max(0.05, float(duration_coeff))
    for note in melody:
        note_duration = float(default_duration)
        note_freq = note

        if isinstance(note, dict):
            note_freq = note.get("freq", note.get("frequency", 0.0))
            if "duration" in note:
                note_duration = float(note["duration"])
        elif isinstance(note, (tuple, list)) and len(note) > 0:
            note_freq = note[0]
            if len(note) > 1:
                note_duration = float(note[1])

        note_duration = max(0.01, note_duration * coeff)
        parsed_notes.append((float(note_freq), note_duration))
    return parsed_notes


def _semitone_distance(freq_a, freq_b):
    if freq_a <= 0.0 or freq_b <= 0.0:
        return 0
    return int(round(12.0 * math.log2(float(freq_b) / float(freq_a))))


def _direction_profile(base_motion, voice_count):
    motion = 1 if base_motion > 0 else (-1 if base_motion < 0 else 0)
    if voice_count == 2:
        return [motion, -motion]
    if voice_count == 3:
        return [motion, 0, -motion]
    if voice_count >= 4:
        return [motion, motion, -motion, -motion]
    return [motion]


def _spread_intervals(voice_count, spread):
    spread = max(0.3, min(5.0, float(spread)))
    if voice_count == 1:
        semitone_offsets = [0.0]
    elif voice_count == 2:
        semitone_offsets = [-5.0 * spread, 5.0 * spread]
    elif voice_count == 3:
        semitone_offsets = [-6.0 * spread, 0.0, 6.0 * spread]
    else:
        semitone_offsets = [-9.0 * spread, -3.0 * spread, 3.0 * spread, 9.0 * spread]
    return [2.0 ** (semitones / 12.0) for semitones in semitone_offsets]


def _build_counterpoint_voices(base_notes, voice_count=1, voice_spread=1.0):
    voice_count = max(1, min(4, int(voice_count)))
    if voice_count <= 1:
        return [list(base_notes)]

    interval_ratios = _spread_intervals(voice_count, voice_spread)
    voices = [[] for _ in range(voice_count)]
    previous_freqs = [0.0] * voice_count

    for note_index, (base_freq, note_duration) in enumerate(base_notes):
        if base_freq <= 0.0:
            for voice_idx in range(voice_count):
                voices[voice_idx].append((0.0, note_duration))
                previous_freqs[voice_idx] = 0.0
            continue

        if note_index == 0:
            for voice_idx in range(voice_count):
                voice_freq = base_freq * interval_ratios[voice_idx]
                voices[voice_idx].append((voice_freq, note_duration))
                previous_freqs[voice_idx] = voice_freq
            continue

        prev_base_freq = base_notes[note_index - 1][0]
        base_motion = 0
        if prev_base_freq > 0.0:
            base_motion = 1 if base_freq > prev_base_freq else (-1 if base_freq < prev_base_freq else 0)

        semitone_step = abs(_semitone_distance(prev_base_freq, base_freq))
        spread = max(0.3, min(5.0, float(voice_spread)))
        semitone_step = max(1, min(7, semitone_step)) if base_motion != 0 else 0
        semitone_step = max(1, int(round(semitone_step * (0.85 + 0.35 * spread)))) if semitone_step > 0 else 0

        direction_by_voice = _direction_profile(base_motion, voice_count)

        for voice_idx in range(voice_count):
            target_freq = base_freq * interval_ratios[voice_idx]
            prev_voice_freq = previous_freqs[voice_idx]

            if prev_voice_freq <= 0.0:
                voice_freq = target_freq
            else:
                direction = direction_by_voice[voice_idx]
                if direction == 0 or semitone_step == 0:
                    moved_freq = prev_voice_freq
                else:
                    moved_freq = prev_voice_freq * (2.0 ** ((direction * semitone_step) / 12.0))

                voice_freq = 0.68 * moved_freq + 0.32 * target_freq

            voice_freq = max(70.0, min(2200.0, voice_freq))
            voices[voice_idx].append((voice_freq, note_duration))
            previous_freqs[voice_idx] = voice_freq

    return voices


def _render_note_sequence(note_sequence, amplitude, waveform):
    def _apply_note_edge_fade(wave_chunk):
        sample_count = wave_chunk.size
        if sample_count <= 4:
            return wave_chunk

        fade_samples = max(8, int(0.0015 * float(_sample_rate)))
        fade_samples = min(fade_samples, sample_count // 2)
        if fade_samples <= 1:
            return wave_chunk

        faded = np.array(wave_chunk, copy=True)
        in_fade = np.linspace(0.0, 1.0, fade_samples, endpoint=False, dtype=np.float32)
        out_fade = np.linspace(1.0, 0.0, fade_samples, endpoint=True, dtype=np.float32)
        faded[:fade_samples] *= in_fade
        faded[-fade_samples:] *= out_fade
        return faded

    chunks = []
    for freq, note_duration in note_sequence:
        if freq <= 0.0:
            sample_count = max(1, int(_sample_rate * note_duration))
            chunks.append(np.zeros(sample_count, dtype=np.float32))
            continue

        raw_wave = _wave_from_freq(freq, note_duration, waveform)
        shaped_wave = _apply_note_edge_fade(raw_wave)
        chunks.append(_build_wave(shaped_wave, amplitude))

    if not chunks:
        return np.zeros(1, dtype=np.float32)
    return np.concatenate(chunks)


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


def generate_melody(
    melody,
    amplitude,
    sample_rate,
    duration_per_note=0.1,
    waveform="sine",
    voices=1,
    counterpoint=True,
    voice_spread=1.0,
    duration_coeff=1.0,
):
    _ensure_audio(sample_rate=sample_rate)
    if melody is None:
        return

    notes = list(melody)
    if not notes:
        return

    parsed_notes = _parse_melody_notes(notes, duration_per_note, duration_coeff=duration_coeff)
    voice_count = max(1, min(4, int(voices)))

    if voice_count == 1 or not counterpoint:
        rendered = _render_note_sequence(parsed_notes, amplitude, waveform)
        _set_current_sound(rendered)
        return

    voice_sequences = _build_counterpoint_voices(parsed_notes, voice_count=voice_count, voice_spread=voice_spread)
    voice_renders = [_render_note_sequence(sequence, amplitude, waveform) for sequence in voice_sequences]

    min_length = min(render.shape[0] for render in voice_renders)
    if min_length <= 0:
        return

    mixed = np.zeros(min_length, dtype=np.float32)
    for render in voice_renders:
        mixed += render[:min_length]

    normalization = max(1.0, float(voice_count) * 0.9)
    mixed /= normalization
    _set_current_sound(mixed)


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
