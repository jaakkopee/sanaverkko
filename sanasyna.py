import math
import random

import numpy as np
import pygame


_sample_rate = 44100
_sound = None
_channel = None


def _ensure_audio(sample_rate=None, buffer_size=1024):
    global _sample_rate

    if sample_rate is not None:
        _sample_rate = int(sample_rate)

    if not pygame.get_init():
        pygame.init()

    if not pygame.mixer.get_init():
        pygame.mixer.pre_init(_sample_rate, -16, 1, buffer_size)
        pygame.mixer.init(_sample_rate, -16, 1, buffer_size)
    else:
        mixer_rate, _, _ = pygame.mixer.get_init()
        _sample_rate = mixer_rate


def _to_int16(samples):
    return np.clip(samples, -32767, 32767).astype(np.int16)


def _create_timebase(duration):
    sample_count = max(1, int(_sample_rate * float(duration)))
    return np.arange(sample_count, dtype=np.float32) / float(_sample_rate)


def _set_current_sound(samples):
    global _sound
    _ensure_audio()
    _sound = pygame.sndarray.make_sound(_to_int16(samples).copy())


def init_sanasyna():
    _ensure_audio()
    return 0


def init_audio(sample_rate=44100):
    _ensure_audio(sample_rate=sample_rate)


def play(loop=True):
    global _channel
    if _sound is None:
        return
    loops = -1 if loop else 0
    _channel = _sound.play(loops=loops)


def stop():
    global _channel
    if _channel is not None:
        _channel.stop()
        _channel = None


def close():
    stop()
    if pygame.mixer.get_init():
        pygame.mixer.quit()


def generate_sine_wave(freq, amplitude, sample_rate, duration=0.25):
    _ensure_audio(sample_rate=sample_rate)
    t = _create_timebase(duration)
    samples = float(amplitude) * np.sin(2.0 * math.pi * float(freq) * t)
    _set_current_sound(samples)


def generate_square_wave(freq, amplitude, sample_rate, duration=0.25):
    _ensure_audio(sample_rate=sample_rate)
    t = _create_timebase(duration)
    samples = float(amplitude) * np.where(np.sin(2.0 * math.pi * float(freq) * t) >= 0.0, 1.0, -1.0)
    _set_current_sound(samples)


def generate_sawtooth_wave(freq, amplitude, sample_rate, duration=0.25):
    _ensure_audio(sample_rate=sample_rate)
    t = _create_timebase(duration)
    phase = (float(freq) * t) % 1.0
    samples = float(amplitude) * (2.0 * phase - 1.0)
    _set_current_sound(samples)


def generate_triangle_wave(freq, amplitude, sample_rate, duration=0.25):
    _ensure_audio(sample_rate=sample_rate)
    t = _create_timebase(duration)
    phase = (float(freq) * t) % 1.0
    samples = float(amplitude) * (4.0 * np.abs(phase - 0.5) - 1.0)
    _set_current_sound(samples)


def generate_noise_wave(freq, amplitude, sample_rate, duration=0.25):
    _ensure_audio(sample_rate=sample_rate)
    _ = freq
    sample_count = max(1, int(_sample_rate * float(duration)))
    samples = float(amplitude) * np.random.uniform(-1.0, 1.0, sample_count)
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
        chunks.append(float(amplitude) * np.sin(2.0 * math.pi * freq * t))

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
    global _sound
    _ensure_audio()
    _sound = buffer


def set_buffer_from_samples(samples, sample_rate, channels):
    _ = channels
    set_buffer(samples, sample_rate)


def set_buffer_from_file(filename):
    global _sound
    _ensure_audio()
    _sound = pygame.mixer.Sound(filename)


def set_sound_buffer_from_file(filename):
    set_buffer_from_file(filename)
