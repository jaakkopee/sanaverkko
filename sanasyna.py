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


def _voice_duration_profile(base_duration, voice_count, note_index, context_size, voice_distance):
    base = max(0.01, float(base_duration))
    count = max(1, int(voice_count))
    context = max(1, int(context_size))
    contour_depth = min(0.22, 0.06 + 0.015 * float(context))
    stability_bias = 1.0 - 0.35 * max(0.0, min(1.0, float(voice_distance)))

    durations = []
    for voice_idx in range(count):
        phase_seed = float(note_index + 1) * (0.72 + 0.09 * float(voice_idx + 1))
        phase = math.sin(phase_seed) + 0.5 * math.sin(0.5 * phase_seed + float(context))
        alternating = -1.0 if (note_index + voice_idx) % 2 == 0 else 1.0
        shape = contour_depth * (0.55 * phase + 0.45 * alternating)
        multiplier = max(0.62, min(1.58, 1.0 + shape * stability_bias))
        durations.append(base * multiplier)

    minimum = max(0.01, base * 0.55)
    maximum = max(minimum + 0.01, base * 1.75)
    return [max(minimum, min(maximum, value)) for value in durations]


def _direction_profile(base_motion, voice_count):
    motion = 1 if base_motion > 0 else (-1 if base_motion < 0 else 0)
    if voice_count == 2:
        return [motion, -motion]
    if voice_count == 3:
        return [motion, 0, -motion]
    if voice_count >= 4:
        return [motion, motion, -motion, -motion]
    return [motion]


def _voice_motion_slopes(voice_count):
    if voice_count == 2:
        return [1.0, 0.8]
    if voice_count == 3:
        return [1.0, 0.55, 0.85]
    if voice_count >= 4:
        return [1.0, 0.8, 0.65, 0.95]
    return [1.0]


def _freq_to_midi(freq):
    if freq <= 0.0:
        return None
    return 69.0 + 12.0 * math.log2(float(freq) / 440.0)


def _midi_to_freq(midi_value):
    return 440.0 * (2.0 ** ((float(midi_value) - 69.0) / 12.0))


def _mode_scale_steps(mapping_mode):
    mode = str(mapping_mode or "original_notes")
    mode_steps = {
        "original_notes": [0.0, 2.0, 4.0, 5.0, 7.0, 9.0, 11.0],
        "pythagorean_pentatonic": [0.0, 2.0, 4.0, 7.0, 9.0],
        "pythagorean_8_note": [0.0, 2.0, 4.0, 5.0, 7.0, 9.0, 11.0],
        "equal_tempered_ionian": [0.0, 2.0, 4.0, 5.0, 7.0, 9.0, 11.0],
        "equal_tempered_dorian": [0.0, 2.0, 3.0, 5.0, 7.0, 9.0, 10.0],
        "equal_tempered_frygian": [0.0, 1.0, 3.0, 5.0, 7.0, 8.0, 10.0],
        "equal_tempered_lydian": [0.0, 2.0, 4.0, 6.0, 7.0, 9.0, 11.0],
        "equal_tempered_mixolydian": [0.0, 2.0, 4.0, 5.0, 7.0, 9.0, 10.0],
        "equal_tempered_aeolian": [0.0, 2.0, 3.0, 5.0, 7.0, 8.0, 10.0],
        "equal_tempered_locrian": [0.0, 1.0, 3.0, 5.0, 6.0, 8.0, 10.0],
    }

    if mode in mode_steps:
        return list(mode_steps[mode])

    if mode.startswith("equal_tempered_") and mode.endswith("_note"):
        parts = mode.split("_")
        try:
            divisions = max(2, int(parts[-2]))
        except Exception:
            divisions = 12
        step_size = 12.0 / float(divisions)
        return [float(index) * step_size for index in range(divisions)]

    return [0.0, 2.0, 4.0, 5.0, 7.0, 9.0, 11.0]


def _nearest_scale_step(value, scale_steps):
    if not scale_steps:
        return 0.0
    return min(scale_steps, key=lambda step: abs(float(step) - float(value)))


def _scale_degree_offset(scale_steps, degree_offset):
    if not scale_steps:
        return 0.0

    ordered = sorted(set(float(step) for step in scale_steps))
    if not ordered:
        ordered = [0.0]

    length = len(ordered)
    octave_shift = int(degree_offset // length)
    local_index = int(degree_offset % length)
    return ordered[local_index] + 12.0 * float(octave_shift)


def _build_scale_interval_library(scale_steps, context_size):
    ordered = sorted(set(float(step) for step in scale_steps))
    if len(ordered) < 2:
        ordered = [0.0, 7.0]

    base_tags = ["baroque", "jazz", "ambient", "folk", "neo-classical", "minimal"]
    fantastic_tags = [
        "quantum-noodle",
        "dragon-telepathy",
        "moon-ladder",
        "time-soup",
        "ghost-polyrhythm",
        "cosmic-potato",
        "hyperbanana-cadence",
        "wormhole-waltz",
    ]

    permutations = []
    interval_names = {
        1: "micro-step", 2: "major-second", 3: "minor-third", 4: "major-third",
        5: "perfect-fourth", 6: "tritone", 7: "perfect-fifth", 8: "minor-sixth",
        9: "major-sixth", 10: "minor-seventh", 11: "major-seventh",
    }
    for source in ordered:
        for target in ordered:
            if abs(source - target) < 1e-6:
                continue
            raw_interval = float(target) - float(source)
            rounded_interval = int(round(raw_interval))
            interval_class = abs(rounded_interval) % 12
            if interval_class == 0:
                continue

            style_tag = base_tags[(interval_class + int(round(source))) % len(base_tags)]
            fantasy_tag = fantastic_tags[(interval_class + int(round(target))) % len(fantastic_tags)]
            permutations.append(
                {
                    "source": source,
                    "target": target,
                    "interval": float(rounded_interval),
                    "name": interval_names.get(interval_class, "mystery-interval"),
                    "tags": [style_tag, fantasy_tag],
                }
            )

    max_step_jump = min(12.0, 2.0 + 1.25 * float(max(1, int(context_size))))
    route_map = {}
    by_interval = {}
    for entry in permutations:
        by_interval.setdefault(int(round(entry["interval"])), []).append(entry)

    unique_intervals = sorted(by_interval.keys())
    for interval in unique_intervals:
        followers = {}
        for candidate in unique_intervals:
            leap = abs(float(candidate) - float(interval))
            if leap > max_step_jump:
                continue
            compatibility = math.exp(-leap / max(1.0, 0.8 * float(max(1, int(context_size)))))
            followers[candidate] = float(compatibility)
        route_map[interval] = followers

    return {
        "entries": permutations,
        "route_map": route_map,
        "by_interval": by_interval,
    }


def _choose_dyad_interval(interval_library, previous_interval, context_size, target_interval):
    by_interval = interval_library.get("by_interval", {})
    route_map = interval_library.get("route_map", {})
    available = sorted(by_interval.keys())
    if not available:
        return float(target_interval)

    if previous_interval is None:
        return float(min(available, key=lambda value: abs(float(value) - float(target_interval))))

    previous_key = int(round(previous_interval))
    followers = route_map.get(previous_key, {})
    if not followers:
        return float(min(available, key=lambda value: abs(float(value) - float(target_interval))))

    best_interval = None
    best_score = None
    complexity = max(1.0, float(max(1, int(context_size))))
    for candidate, weight in followers.items():
        closeness = abs(float(candidate) - float(target_interval))
        static_penalty = 1.5 if abs(float(candidate)) < 0.5 else 0.0
        score = (1.0 - min(0.85, 0.1 * complexity)) * closeness - 1.2 * float(weight) + static_penalty
        if best_score is None or score < best_score:
            best_score = score
            best_interval = candidate

    if best_interval is None:
        best_interval = min(available, key=lambda value: abs(float(value) - float(target_interval)))
    return float(best_interval)


def _build_chord_targets(
    base_midi,
    voice_count,
    spread,
    scale_steps,
    interval_library,
    previous_midis,
    dominant_state,
    context_size,
):
    if base_midi is None:
        return [None] * voice_count, None

    chroma = float(base_midi) % 12.0
    nearest_step = _nearest_scale_step(chroma, scale_steps)
    root_midi = float(base_midi) - chroma + nearest_step

    urgency = float(dominant_state.get("urgency", 0.0))
    target_root = dominant_state.get("target_root")
    if target_root is not None and urgency > 0.05:
        root_midi = (1.0 - urgency) * root_midi + urgency * float(target_root)

    if voice_count <= 1:
        return [root_midi], root_midi

    if voice_count == 2:
        preferred_interval = max(2.0, 5.0 * float(spread))
        previous_interval = None
        if previous_midis and len(previous_midis) >= 2 and previous_midis[0] is not None and previous_midis[1] is not None:
            previous_interval = float(previous_midis[1] - previous_midis[0])

        selected_interval = _choose_dyad_interval(
            interval_library=interval_library,
            previous_interval=previous_interval,
            context_size=context_size,
            target_interval=preferred_interval,
        )
        return [root_midi, root_midi + selected_interval], root_midi

    triad_degrees = [0, 2, 4]
    if voice_count >= 4:
        triad_degrees.append(6)

    target_midis = []
    for degree in triad_degrees[:voice_count]:
        chord_offset = _scale_degree_offset(scale_steps, degree)
        target_midis.append(root_midi + chord_offset)

    for index in range(1, len(target_midis)):
        while target_midis[index] <= target_midis[index - 1] + 1.0:
            target_midis[index] += 12.0

    return target_midis, root_midi


def _update_dominant_state(dominant_state, root_midi, context_size):
    decay = math.exp(-1.0 / max(1.0, float(max(1, int(context_size)))))
    urgency = float(dominant_state.get("urgency", 0.0)) * decay

    target_root = dominant_state.get("target_root")
    if target_root is not None and root_midi is not None:
        if abs(float(root_midi) - float(target_root)) <= 1.5:
            urgency *= 0.35

    next_target = None
    if root_midi is not None:
        next_target = float(root_midi) - 7.0

    promoted_urgency = max(urgency, 0.45 + 0.45 * (1.0 - decay))
    dominant_state["urgency"] = max(0.0, min(0.92, promoted_urgency))
    dominant_state["target_root"] = next_target


def _motion_sign(current_value, previous_value):
    if current_value > previous_value:
        return 1
    if current_value < previous_value:
        return -1
    return 0


def _is_perfect_interval(interval_semitones):
    interval_class = abs(int(round(interval_semitones))) % 12
    return interval_class in (0, 7)


def _voice_distance_cost(candidate_midi, history_midis, context_size):
    if candidate_midi is None:
        return 0.0

    recent = [value for value in history_midis[-max(1, int(context_size)):] if value is not None]
    if not recent:
        return 0.0

    weighted_sum = 0.0
    total_weight = 0.0
    for index, midi_value in enumerate(reversed(recent)):
        weight = float(index + 1)
        weighted_sum += weight * abs(float(candidate_midi) - float(midi_value))
        total_weight += weight

    average_distance = weighted_sum / max(1.0, total_weight)
    immediate_step = abs(float(candidate_midi) - float(recent[-1]))
    return 0.55 * immediate_step + 0.45 * average_distance


def _choose_voice_candidate(
    target_midi,
    moved_midi,
    previous_history,
    base_midi,
    max_register_distance,
    voice_distance,
    voice_distance_context,
):
    if target_midi is None:
        return None

    primary = moved_midi if moved_midi is not None else target_midi
    center = (1.0 - voice_distance) * target_midi + voice_distance * primary

    best_midi = None
    best_score = None

    for semitone_offset in range(-8, 9):
        candidate = center + float(semitone_offset)

        if base_midi is not None:
            while candidate - base_midi > max_register_distance:
                candidate -= 12.0
            while base_midi - candidate > max_register_distance:
                candidate += 12.0

        target_cost = abs(candidate - target_midi)
        path_cost = _voice_distance_cost(candidate, previous_history, voice_distance_context)
        score = (1.0 - voice_distance) * target_cost + voice_distance * path_cost

        if best_score is None or score < best_score:
            best_score = score
            best_midi = candidate

    return best_midi


def _avoid_parallel_perfects(candidate_midis, previous_midis):
    if not candidate_midis or not previous_midis:
        return candidate_midis

    adjusted = list(candidate_midis)
    max_iterations = 3
    for iteration in range(max_iterations):
        changed = False
        for lower_index in range(len(adjusted) - 1):
            for upper_index in range(lower_index + 1, len(adjusted)):
                prev_low = previous_midis[lower_index]
                prev_high = previous_midis[upper_index]
                curr_low = adjusted[lower_index]
                curr_high = adjusted[upper_index]

                if prev_low is None or prev_high is None or curr_low is None or curr_high is None:
                    continue

                low_motion = _motion_sign(curr_low, prev_low)
                high_motion = _motion_sign(curr_high, prev_high)
                if low_motion == 0 or high_motion == 0 or low_motion != high_motion:
                    continue

                prev_interval = prev_high - prev_low
                curr_interval = curr_high - curr_low
                if not (_is_perfect_interval(prev_interval) and _is_perfect_interval(curr_interval)):
                    continue

                moved_direction = high_motion
                candidate_offsets = [1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7]
                if moved_direction < 0:
                    candidate_offsets = [-1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6, -7, 7]

                best_proposal = None
                best_distance = float('inf')
                for offset in candidate_offsets:
                    proposal = curr_high + offset
                    if proposal - curr_low < 2.0:
                        continue
                    proposed_interval = proposal - curr_low
                    if _is_perfect_interval(proposed_interval):
                        continue
                    distance_from_original = abs(offset)
                    if distance_from_original < best_distance:
                        best_distance = distance_from_original
                        best_proposal = proposal

                if best_proposal is not None:
                    adjusted[upper_index] = best_proposal
                    changed = True

        if not changed:
            break

    return adjusted


def _avoid_hidden_perfects(candidate_midis, previous_midis):
    if not candidate_midis or not previous_midis:
        return candidate_midis

    adjusted = list(candidate_midis)
    max_iterations = 3
    for iteration in range(max_iterations):
        changed = False
        for lower_index in range(len(adjusted) - 1):
            for upper_index in range(lower_index + 1, len(adjusted)):
                prev_low = previous_midis[lower_index]
                prev_high = previous_midis[upper_index]
                curr_low = adjusted[lower_index]
                curr_high = adjusted[upper_index]

                if prev_low is None or prev_high is None or curr_low is None or curr_high is None:
                    continue

                low_motion = _motion_sign(curr_low, prev_low)
                high_motion = _motion_sign(curr_high, prev_high)
                if low_motion == 0 or high_motion == 0 or low_motion != high_motion:
                    continue

                current_interval = curr_high - curr_low
                if not _is_perfect_interval(current_interval):
                    continue

                upper_leap = abs(curr_high - prev_high)
                lower_leap = abs(curr_low - prev_low)
                if upper_leap <= 2.0 and lower_leap <= 2.0:
                    continue

                preferred_offsets = [1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7]
                if high_motion < 0:
                    preferred_offsets = [-1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6, -7, 7]

                best_proposal = None
                best_distance = float('inf')
                for offset in preferred_offsets:
                    proposal = curr_high + offset
                    if proposal - curr_low < 2.0:
                        continue
                    if _is_perfect_interval(proposal - curr_low):
                        continue
                    distance_from_original = abs(offset)
                    if distance_from_original < best_distance:
                        best_distance = distance_from_original
                        best_proposal = proposal

                if best_proposal is not None:
                    adjusted[upper_index] = best_proposal
                    changed = True

        if not changed:
            break

    return adjusted


def _spread_intervals(voice_count, spread):
    spread = max(0.3, min(5.0, float(spread)))
    if voice_count == 1:
        semitone_offsets = [0.0]
    elif voice_count == 2:
        semitone_offsets = [-3.0 * spread, 3.0 * spread]
    elif voice_count == 3:
        semitone_offsets = [-5.0 * spread, 0.0, 5.0 * spread]
    else:
        semitone_offsets = [-7.0 * spread, -2.0 * spread, 2.0 * spread, 7.0 * spread]
    return [2.0 ** (semitones / 12.0) for semitones in semitone_offsets]


def _build_counterpoint_voices(
    base_notes,
    voice_count=1,
    voice_spread=1.0,
    strict_counterpoint=False,
    voice_distance=0.65,
    voice_distance_context=4,
    mapping_mode="original_notes",
):
    voice_count = max(1, min(4, int(voice_count)))
    if voice_count <= 1:
        return [list(base_notes)]

    interval_ratios = _spread_intervals(voice_count, voice_spread)
    scale_steps = _mode_scale_steps(mapping_mode)
    interval_library = _build_scale_interval_library(scale_steps, voice_distance_context)
    voice_distance = min(1.0, max(0.0, float(voice_distance)))
    voice_distance_context = max(1, int(voice_distance_context))

    voices = [[] for _ in range(voice_count)]
    previous_freqs = [0.0] * voice_count
    previous_midis = [None] * voice_count
    voice_histories = [[] for _ in range(voice_count)]
    slope_profile = _voice_motion_slopes(voice_count)
    dominant_state = {"urgency": 0.0, "target_root": None}

    for note_index, (base_freq, note_duration) in enumerate(base_notes):
        per_voice_durations = _voice_duration_profile(
            base_duration=note_duration,
            voice_count=voice_count,
            note_index=note_index,
            context_size=voice_distance_context,
            voice_distance=voice_distance,
        )

        if base_freq <= 0.0:
            for voice_idx in range(voice_count):
                voices[voice_idx].append((0.0, per_voice_durations[voice_idx]))
                previous_freqs[voice_idx] = 0.0
                previous_midis[voice_idx] = None
                voice_histories[voice_idx].append(None)
            continue

        base_midi = _freq_to_midi(base_freq)
        routed_target_midis, routed_root_midi = _build_chord_targets(
            base_midi=base_midi,
            voice_count=voice_count,
            spread=voice_spread,
            scale_steps=scale_steps,
            interval_library=interval_library,
            previous_midis=previous_midis,
            dominant_state=dominant_state,
            context_size=voice_distance_context,
        )
        target_midis = []
        for voice_idx in range(voice_count):
            if voice_idx < len(routed_target_midis) and routed_target_midis[voice_idx] is not None:
                target_midis.append(routed_target_midis[voice_idx])
            else:
                target_freq = base_freq * interval_ratios[voice_idx]
                target_midis.append(_freq_to_midi(target_freq))

        if note_index == 0:
            for voice_idx in range(voice_count):
                voice_freq = _midi_to_freq(target_midis[voice_idx])
                voices[voice_idx].append((voice_freq, per_voice_durations[voice_idx]))
                previous_freqs[voice_idx] = voice_freq
                previous_midis[voice_idx] = target_midis[voice_idx]
                voice_histories[voice_idx].append(target_midis[voice_idx])
            _update_dominant_state(dominant_state, routed_root_midi, voice_distance_context)
            continue

        prev_base_freq = base_notes[note_index - 1][0]
        base_motion = 0
        if prev_base_freq > 0.0:
            base_motion = 1 if base_freq > prev_base_freq else (-1 if base_freq < prev_base_freq else 0)

        semitone_step = abs(_semitone_distance(prev_base_freq, base_freq))
        spread = max(0.3, min(5.0, float(voice_spread)))
        semitone_step = max(1, min(4, semitone_step)) if base_motion != 0 else 0
        semitone_step = max(1, int(round(semitone_step * (0.7 + 0.15 * spread)))) if semitone_step > 0 else 0

        direction_by_voice = _direction_profile(base_motion, voice_count)
        candidate_midis = [None] * voice_count

        for voice_idx in range(voice_count):
            target_midi = target_midis[voice_idx]
            prev_voice_midi = previous_midis[voice_idx]

            if prev_voice_midi is None:
                voice_midi = target_midi
            else:
                direction = direction_by_voice[voice_idx]
                if direction == 0 or semitone_step == 0:
                    moved_midi = prev_voice_midi
                else:
                    slope = slope_profile[voice_idx]
                    local_step = max(1, int(round(float(semitone_step) * float(slope))))
                    moved_midi = prev_voice_midi + direction * local_step

                max_register_distance = 18.0 + 4.0 * spread
                voice_midi = _choose_voice_candidate(
                    target_midi=target_midi,
                    moved_midi=moved_midi,
                    previous_history=voice_histories[voice_idx],
                    base_midi=base_midi,
                    max_register_distance=max_register_distance,
                    voice_distance=voice_distance,
                    voice_distance_context=voice_distance_context,
                )

            if voice_midi is not None and base_midi is not None:
                max_register_distance = 18.0 + 4.0 * spread
                while voice_midi - base_midi > max_register_distance:
                    voice_midi -= 12.0
                while base_midi - voice_midi > max_register_distance:
                    voice_midi += 12.0

            candidate_midis[voice_idx] = voice_midi

        for voice_idx in range(1, voice_count):
            if candidate_midis[voice_idx] is not None and candidate_midis[voice_idx - 1] is not None:
                min_spacing = 2.0
                if candidate_midis[voice_idx] - candidate_midis[voice_idx - 1] < min_spacing:
                    candidate_midis[voice_idx] = candidate_midis[voice_idx - 1] + min_spacing

        candidate_midis = _avoid_parallel_perfects(candidate_midis, previous_midis)
        if strict_counterpoint:
            candidate_midis = _avoid_hidden_perfects(candidate_midis, previous_midis)
            candidate_midis = _avoid_parallel_perfects(candidate_midis, previous_midis)

        if note_index >= 1:
            static_voices = 0
            for voice_idx in range(voice_count):
                previous_midi = previous_midis[voice_idx]
                current_midi = candidate_midis[voice_idx]
                if previous_midi is None or current_midi is None:
                    continue
                if abs(float(current_midi) - float(previous_midi)) < 0.2:
                    static_voices += 1

            if static_voices >= max(1, voice_count - 1):
                for voice_idx in range(voice_count - 1, 0, -1):
                    if candidate_midis[voice_idx] is None or candidate_midis[voice_idx - 1] is None:
                        continue
                    nudge = 1.0 if (note_index + voice_idx) % 2 == 0 else -1.0
                    proposal = candidate_midis[voice_idx] + nudge
                    if proposal - candidate_midis[voice_idx - 1] >= 2.0:
                        candidate_midis[voice_idx] = proposal
                        break

        for voice_idx in range(voice_count):
            voice_midi = candidate_midis[voice_idx]
            if voice_midi is None:
                voice_freq = 0.0
            else:
                voice_freq = _midi_to_freq(voice_midi)

            voice_freq = max(70.0, min(2200.0, voice_freq))
            voices[voice_idx].append((voice_freq, per_voice_durations[voice_idx]))
            previous_freqs[voice_idx] = voice_freq
            stored_midi = _freq_to_midi(voice_freq)
            previous_midis[voice_idx] = stored_midi
            voice_histories[voice_idx].append(stored_midi)
            if len(voice_histories[voice_idx]) > max(4, voice_distance_context * 2):
                voice_histories[voice_idx] = voice_histories[voice_idx][-max(4, voice_distance_context * 2):]

        _update_dominant_state(dominant_state, routed_root_midi, voice_distance_context)

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
    strict_counterpoint=False,
    voice_spread=1.0,
    voice_distance=0.65,
    voice_distance_context=4,
    mapping_mode="original_notes",
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

    voice_sequences = _build_counterpoint_voices(
        parsed_notes,
        voice_count=voice_count,
        voice_spread=voice_spread,
        strict_counterpoint=bool(strict_counterpoint),
        voice_distance=voice_distance,
        voice_distance_context=voice_distance_context,
        mapping_mode=mapping_mode,
    )
    voice_renders = [_render_note_sequence(sequence, amplitude, waveform) for sequence in voice_sequences]

    max_length = max(render.shape[0] for render in voice_renders)
    if max_length <= 0:
        return

    mixed = np.zeros(max_length, dtype=np.float32)
    for render in voice_renders:
        local_length = render.shape[0]
        mixed[:local_length] += render

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
