import numpy as np
import pygame
import random
import time
import math
import bisect
import wx
import threading
import sys
import os
import json
import shutil
import subprocess
import tempfile
import wave
from collections import deque
import sanasyna

try:
    import sounddevice as sd
except Exception:
    sd = None

try:
    import sv_ltm
except Exception:
    sv_ltm = None

try:
    import nltk
except Exception:
    nltk = None

try:
    import pygame.font as pygame_font
except Exception:
    pygame_font = None

try:
    import pygame._sdl2.video as pygame_sdl2_video
except Exception:
    pygame_sdl2_video = None


MONOSPACE_FONT_CANDIDATES = [
    "JetBrains Mono",
    "Fira Code",
    "IBM Plex Mono",
    "Consolas",
    "Menlo",
    "Monaco",
    "Courier New",
    "monospace",
]

_pygame_mono_font_cache = {}



gematria_table = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7, "h": 8, "i": 9, "j": 10, "k": 20, "l": 30, "m": 40, "n": 50, "o": 60, "p": 70, "q": 80, "r": 90, "s": 100, "t": 200, "u": 300, "v": 400, "w": 500, "x": 600, "y": 700, "z": 800, "å": 900, "ä": 1000, "ö": 1100}


class ADSRDisplayPanel(wx.Panel):
    def __init__(self, parent):
        super().__init__(parent, size=(240, 90))
        self.attack = 0.01
        self.decay = 0.04
        self.sustain = 0.85
        self.release = 0.03
        self.Bind(wx.EVT_PAINT, self.OnPaint)

    def set_adsr(self, attack, decay, sustain, release):
        self.attack = max(0.0, float(attack))
        self.decay = max(0.0, float(decay))
        self.sustain = min(1.0, max(0.0, float(sustain)))
        self.release = max(0.0, float(release))
        self.Refresh()

    def OnPaint(self, event):
        _ = event
        dc = wx.PaintDC(self)
        width, height = self.GetClientSize()

        dc.SetBrush(wx.Brush(wx.Colour(30, 30, 30)))
        dc.SetPen(wx.Pen(wx.Colour(60, 60, 60), 1))
        dc.DrawRectangle(0, 0, width, height)

        left = 8
        top = 8
        draw_width = max(10, width - 16)
        draw_height = max(10, height - 16)
        bottom = top + draw_height

        total_time = self.attack + self.decay + self.release + 0.25
        if total_time <= 0:
            total_time = 1.0

        def x_from_time(time_value):
            return left + int((time_value / total_time) * draw_width)

        def y_from_level(level_value):
            level = min(1.0, max(0.0, float(level_value)))
            return top + int((1.0 - level) * draw_height)

        t0 = 0.0
        t1 = self.attack
        t2 = t1 + self.decay
        t3 = total_time - self.release
        t4 = total_time

        points = [
            (x_from_time(t0), y_from_level(0.0)),
            (x_from_time(t1), y_from_level(1.0)),
            (x_from_time(t2), y_from_level(self.sustain)),
            (x_from_time(t3), y_from_level(self.sustain)),
            (x_from_time(t4), y_from_level(0.0)),
        ]

        dc.SetPen(wx.Pen(wx.Colour(180, 220, 255), 2))
        for index in range(len(points) - 1):
            x1, y1 = points[index]
            x2, y2 = points[index + 1]
            dc.DrawLine(x1, y1, x2, y2)

        dc.SetPen(wx.Pen(wx.Colour(90, 90, 90), 1))
        dc.DrawLine(left, bottom, left + draw_width, bottom)
        dc.DrawLine(left, top, left, bottom)


class RhythmTimelinePreviewPanel(wx.Panel):
    def __init__(self, parent):
        super().__init__(parent, size=(920, 120))
        self.blocks = []
        self.signature = "4/4"
        self.additive_weight = 0.0
        self.divisive_weight = 0.0
        self.Bind(wx.EVT_PAINT, self.OnPaint)

    def set_rhythm_state(self, blocks, signature, additive_weight, divisive_weight):
        normalized = []
        if isinstance(blocks, (list, tuple)):
            for value in blocks:
                try:
                    pulse = int(float(value))
                except Exception:
                    continue
                if pulse > 0:
                    normalized.append(pulse)
        self.blocks = normalized
        self.signature = str(signature or "4/4")
        self.additive_weight = min(1.0, max(0.0, float(additive_weight)))
        self.divisive_weight = min(1.0, max(0.0, float(divisive_weight)))
        self.Refresh()

    def _signature_parts(self):
        num = 4
        den = 4
        if "/" in self.signature:
            parts = self.signature.split("/", 1)
            try:
                num = max(1, int(parts[0]))
                den = int(parts[1])
            except Exception:
                num, den = 4, 4
        if den not in {2, 4, 8, 16}:
            den = 4
        return num, den

    def OnPaint(self, event):
        _ = event
        dc = wx.PaintDC(self)
        width, height = self.GetClientSize()

        dc.SetBrush(wx.Brush(wx.Colour(24, 24, 24)))
        dc.SetPen(wx.Pen(wx.Colour(52, 52, 52), 1))
        dc.DrawRectangle(0, 0, width, height)

        left = 8
        right = max(left + 10, width - 8)
        top = 8
        bottom = max(top + 20, height - 8)
        usable_width = max(20, right - left)
        usable_height = max(20, bottom - top)

        blocks = list(self.blocks)
        if not blocks:
            dc.SetTextForeground(wx.Colour(210, 210, 210))
            dc.DrawText("Timeline preview  Additive blocks: (empty chain)  Signature: " + self.signature, left + 4, top + 2)
            dc.SetTextForeground(wx.Colour(180, 180, 180))
            dc.DrawText("Append blocks from the palette to generate BPM-linked bangs.", left + 4, top + 22)
            return
        total_pulses = max(1, sum(blocks))
        slots = max(96, min(512, total_pulses * 24))
        slot_width = usable_width / float(slots)
        pulse_positions = np.linspace(0.0, float(total_pulses), num=slots, endpoint=False, dtype=np.float64)

        onset_positions = np.cumsum(np.array([0] + blocks[:-1], dtype=np.float64))
        distance = np.full(slots, np.inf, dtype=np.float64)
        for onset in onset_positions:
            local = np.mod(pulse_positions - onset + float(total_pulses), float(total_pulses))
            distance = np.minimum(distance, local)
        mean_block = max(1.0, float(total_pulses) / float(len(blocks)))
        additive_decay = max(0.12, 0.24 * mean_block)
        additive_env = (0.30 + 0.70 * np.exp(-distance / additive_decay)).astype(np.float32)

        num, den = self._signature_parts()
        beat_pulses = max(0.5, 4.0 / float(den))
        bar_pulses = beat_pulses * float(num)
        bar_pos = np.mod(pulse_positions, bar_pulses)
        beat_index = np.floor(bar_pos / beat_pulses)
        beat_distance = np.mod(bar_pos, beat_pulses)
        downbeat = np.exp(-beat_distance / max(0.08, 0.20 * beat_pulses))
        beat = np.exp(-beat_distance / max(0.08, 0.14 * beat_pulses))
        divisive_env = np.where(beat_index == 0.0, 0.30 + 0.70 * downbeat, 0.30 + 0.40 * beat).astype(np.float32)

        combined = ((1.0 - self.additive_weight) + self.additive_weight * additive_env) * (
            (1.0 - self.divisive_weight) + self.divisive_weight * divisive_env
        )
        combined = np.clip(combined, 0.0, 1.0)

        dc.SetPen(wx.Pen(wx.Colour(70, 70, 70), 1))
        dc.DrawLine(left, bottom - 1, right, bottom - 1)
        dc.DrawLine(left, top, left, bottom)

        for pulse_idx, gain in enumerate(combined):
            x0 = int(left + pulse_idx * slot_width)
            x1 = int(left + (pulse_idx + 1) * slot_width)
            if x1 <= x0:
                x1 = x0 + 1

            height_ratio = max(0.05, gain)
            bar_height = int(height_ratio * (usable_height - 18))
            y0 = bottom - bar_height - 2

            add_val = float(additive_env[pulse_idx])
            div_val = float(divisive_env[pulse_idx])
            red = int(70 + 120 * add_val)
            green = int(70 + 120 * div_val)
            blue = int(95 + 110 * float(gain))

            dc.SetBrush(wx.Brush(wx.Colour(red, green, blue)))
            dc.SetPen(wx.Pen(wx.Colour(25, 25, 25), 1))
            dc.DrawRectangle(x0, y0, max(1, x1 - x0), bar_height)

        # Draw block-start markers to emphasize additive IOIs
        dc.SetPen(wx.Pen(wx.Colour(230, 210, 90), 1))
        running = 0.0
        for block in blocks:
            x = int(left + (running / float(total_pulses)) * usable_width)
            dc.DrawLine(x, top + 14, x, bottom - 2)
            running += float(block)

        dc.SetTextForeground(wx.Colour(210, 210, 210))
        info = f"Timeline preview  Additive blocks: {'+'.join(str(b) for b in blocks)}  Signature: {num}/{den}"
        dc.DrawText(info, left + 4, top + 2)

class SanaVerkkoKontrolleri:
    
    def __init__(self):
        self.params = {}
        self.params["set_weight_by_gematria"] = False
        self.params["use_pos_matching"] = False
        self.params["fluid_pos"] = False
        self.params["use_long_term_memory"] = False
        self.params["ltm_weight"] = 0.35
        self.params["common_word_penalty"] = True
        self.params["learning_rate"] = 0.1
        self.params["error"] = 0
        self.params["target"] = 0
        self.params["activation_increase"] = 0.0001
        self.params["activation_limit"] = 2
        self.params["sigmoid_scale"] = 2
        self.params["word_change_threshold"] = 0.777
        self.params["zoom"]=0.1
        self.params["process_interval"] = 0.1
        self.params["process_interval_from_rhythm_bpm"] = False
        self.params["logic_iteration_limit"] = 48
        self.params["selection_exploration"] = 0.18
        self.params["selection_top_k"] = 4
        self.params["jump_probability"] = 0.08
        self.params["jump_radius"] = 120
        self.params["use_phoneme_rhyme"] = True
        self.params["rhyme_weight"] = 0.28
        self.params["rhyme_min_similarity"] = 0.34
        self.params["rhyme_strategy"] = "hybrid"
        self.params["rhyme_tail_bias"] = 0.60
        self.params["fluid_root"] = False
        self.params["fluid_gematria"] = False
        self.params["import_mode"] = "append"
        self.params["audio_wave_mode"] = "dynamic"
        self.params["frequency_mapping_mode"] = "original_notes"
        self.params["voice_count"] = 1
        self.params["voice_spread"] = 1.0
        self.params["voice_distance"] = 0.65
        self.params["voice_distance_context"] = 4
        self.params["rhythmic_divergence"] = 0.35
        self.params["rhythm_style"] = "manual"
        self.params["beat_library_style"] = "auto"
        self.params["rhythm_gate_strength"] = 0.85
        self.params["rhythm_stretch_strength"] = 1.0
        self.params["rhythm_rotation"] = 0
        self.params["rhythm_radicality"] = 0.5
        self.params["rhythm_mod_bpm"] = 108.0
        self.params["additive_rhythm_blocks"] = []
        self.params["additive_rhythm_weight"] = 0.0
        self.params["divisive_rhythm_signature"] = "4/4"
        self.params["divisive_rhythm_weight"] = 0.0
        self.params["strict_counterpoint"] = True
        self.params["melody_coherence"] = 0.65
        self.params["melody_speed"] = 1.0
        self.params["min_note_duration"] = 0.03
        self.params["melody_from_own_time"] = True
        self.params["piper_tts_on"] = False
        self.params["piper_model_path"] = os.environ.get("PIPER_MODEL", "")
        self.params["piper_volume"] = 0.5
        self.params["synth_volume"] = 1.0
        self.params["compressor_enabled"] = False
        self.params["compressor_threshold_db"] = -18.0
        self.params["compressor_ratio"] = 3.0
        self.params["compressor_makeup_db"] = 6.0
        self.params["fullscreen"] = False
        self.params["adsr_attack"] = 0.01
        self.params["adsr_decay"] = 0.04
        self.params["adsr_sustain"] = 0.85
        self.params["adsr_release"] = 0.03

        self.app = wx.App()

        self.frame = wx.Frame(None, -1, "SanaVerkko")
        self.frame.Bind(wx.EVT_CLOSE, self.OnClose)
        self.frame.SetSize(760, 900)
        self.words = []
        self.referenceWords = []
        self.wordsToChange = []
        self.wordsToChangeIndex = 0
        self.running = True
        self.closed = False
        self.timer = None

        self.screen = None
        self.size = None
        self.clock = None
        self._windowed_size = (1024, 768)
        self._pygame_window = None

        self.conn_color_r = 0
        self.conn_color_g = 0
        self.conn_color_b = 0
        self.last_process_time = 0
        self.last_result_sentence = ""
        self.last_result_gematria_line = ""
        self.last_result_reduction_line = ""
        self.output_file_path = os.path.abspath("output.txt")
        self.output_frame = None
        self.output_text_ctrl = None
        self.output_timer = None
        self.last_output_content = ""
        self.pos_tag_cache = {}
        self.nltk_pos_ready = False
        self.nltk_pos_init_attempted = False
        self.reference_index_dirty = True
        self.reference_index_has_pos = False
        self.reference_index = {}
        self.phoneme_cache = {}
        self._last_param_event = {}
        self._suppress_param_events = False
        self._in_param_commit = False
        self.ltm_model = None
        self.ltm_model_path = ""
        self.ltm_context_size = 3
        self._logic_thread = None

        self._logic_state_lock = threading.Lock()
        self._logic_inflight = False
        self._logic_changed_ready = False
        self._piper_speak_thread = None
        self._last_spoken_sentence = ""
        self._last_queued_sentence = ""
        self._piper_sentence_queue = deque()
        self._piper_queue_lock = threading.Lock()
        self._piper_queue_event = threading.Event()
        self._piper_worker_stop = False
        self._piper_audio_channel = None
        self._piper_audio_sound = None
        self.additive_editor_frame = None
        self.additive_sequence_scroll = None
        self.additive_sequence_sizer = None
        self.additive_timeline_preview = None

        self.initPygame()
        self.initAudio()
        self.initWords()
        self.outfile = open(self.output_file_path, "w")
        self.widgetSetup()

    def widgetSetup(self):
        panel = wx.ScrolledWindow(self.frame, -1, style=wx.VSCROLL)
        panel.SetScrollRate(10, 10)
        self.set_weight_by_gematria_checkbox = wx.CheckBox(panel, -1, "Set weight by gematria")
        self.set_weight_by_gematria_checkbox.SetValue(self.params["set_weight_by_gematria"])
        self.set_weight_by_gematria_checkbox.Bind(wx.EVT_CHECKBOX, self.OnSetWeightByGematria)

        self.use_pos_matching_checkbox = wx.CheckBox(panel, -1, "Use POS matching")
        self.use_pos_matching_checkbox.SetValue(self.params["use_pos_matching"])
        self.use_pos_matching_checkbox.Bind(wx.EVT_CHECKBOX, self.OnUsePOSMatching)
        self.pos_backend_status_label = wx.StaticText(panel, -1, "POS backend: heuristic")

        self.fluid_pos_checkbox = wx.CheckBox(panel, -1, "Fluid POS")
        self.fluid_pos_checkbox.SetValue(self.params["fluid_pos"])
        self.fluid_pos_checkbox.Bind(wx.EVT_CHECKBOX, self.OnFluidPOS)

        self.use_long_term_memory_checkbox = wx.CheckBox(panel, -1, "Use long term memory")
        self.use_long_term_memory_checkbox.SetValue(self.params["use_long_term_memory"])
        self.use_long_term_memory_checkbox.Bind(wx.EVT_CHECKBOX, self.OnUseLongTermMemory)

        self.ltm_weight_label = wx.StaticText(panel, -1, "LTM weight (0-1)")
        self.ltm_weight_ctrl = wx.TextCtrl(panel, -1, str(self.params["ltm_weight"]), style=wx.TE_PROCESS_ENTER)
        self._bindNumericCtrl(self.ltm_weight_ctrl, self.OnLTMWeight)

        self.common_word_penalty_checkbox = wx.CheckBox(panel, -1, "Common word penalty")
        self.common_word_penalty_checkbox.SetValue(self.params["common_word_penalty"])
        self.common_word_penalty_checkbox.Bind(wx.EVT_CHECKBOX, self.OnCommonWordPenalty)

        self.ltm_load_button = wx.Button(panel, -1, "Load long term memory")
        self.ltm_load_button.Bind(wx.EVT_BUTTON, self.OnLoadLongTermMemory)
        self.ltm_status_label = wx.StaticText(panel, -1, "LTM: not loaded")

        self.fluid_root_checkbox = wx.CheckBox(panel, -1, "Fluid root")
        self.fluid_root_checkbox.SetValue(self.params["fluid_root"])
        self.fluid_root_checkbox.Bind(wx.EVT_CHECKBOX, self.OnFluidRoot)

        self.fluid_gematria_checkbox = wx.CheckBox(panel, -1, "Fluid gematria")
        self.fluid_gematria_checkbox.SetValue(self.params["fluid_gematria"])
        self.fluid_gematria_checkbox.Bind(wx.EVT_CHECKBOX, self.OnFluidGematria)

        self.learning_rate_label = wx.StaticText(panel, -1, "Learning rate")
        self.learning_rate_ctrl = wx.TextCtrl(panel, -1, str(self.params["learning_rate"]), style=wx.TE_PROCESS_ENTER)
        self._bindNumericCtrl(self.learning_rate_ctrl, self.OnLearningRate)

        self.error_label = wx.StaticText(panel, -1, "Error")
        self.error_ctrl = wx.TextCtrl(panel, -1, str(self.params["error"]), style=wx.TE_PROCESS_ENTER)
        self._bindNumericCtrl(self.error_ctrl, self.OnError)

        self.activation_increase_label = wx.StaticText(panel, -1, "Activation increase")
        self.activation_increase_ctrl = wx.TextCtrl(panel, -1, str(self.params["activation_increase"]), style=wx.TE_PROCESS_ENTER)
        self._bindNumericCtrl(self.activation_increase_ctrl, self.OnActivationIncrease)

        self.activation_limit_label = wx.StaticText(panel, -1, "Activation limit")
        self.activation_limit_ctrl = wx.TextCtrl(panel, -1, str(self.params["activation_limit"]), style=wx.TE_PROCESS_ENTER)
        self._bindNumericCtrl(self.activation_limit_ctrl, self.OnActivationLimit)

        self.sigmoid_scale_label = wx.StaticText(panel, -1, "Sigmoid scale")
        self.sigmoid_scale_ctrl = wx.TextCtrl(panel, -1, str(self.params["sigmoid_scale"]), style=wx.TE_PROCESS_ENTER)
        self._bindNumericCtrl(self.sigmoid_scale_ctrl, self.OnSigmoidScale)

        self.word_change_threshold_label = wx.StaticText(panel, -1, "Word change threshold")
        self.word_change_threshold_ctrl = wx.TextCtrl(panel, -1, str(self.params["word_change_threshold"]), style=wx.TE_PROCESS_ENTER)
        self._bindNumericCtrl(self.word_change_threshold_ctrl, self.OnWordChangeThreshold)

        self.zoom_label = wx.StaticText(panel, -1, "Zoom")
        self.zoom_ctrl = wx.TextCtrl(panel, -1, str(self.params["zoom"]), style=wx.TE_PROCESS_ENTER)
        self._bindNumericCtrl(self.zoom_ctrl, self.OnZoom)

        self.process_interval_label = wx.StaticText(panel, -1, "Process interval (s)")
        self.process_interval_ctrl = wx.TextCtrl(panel, -1, str(self.params["process_interval"]), style=wx.TE_PROCESS_ENTER)
        self._bindNumericCtrl(self.process_interval_ctrl, self.OnProcessInterval)
        self.process_interval_bpm_sync_checkbox = wx.CheckBox(panel, -1, "Sync to rhythm BPM")
        self.process_interval_bpm_sync_checkbox.SetValue(bool(self.params.get("process_interval_from_rhythm_bpm", False)))
        self.process_interval_bpm_sync_checkbox.Bind(wx.EVT_CHECKBOX, self.OnProcessIntervalBPMMode)
        self.process_interval_value_label = wx.StaticText(panel, -1, "Current interval: 0.100 s (manual)")

        self.melody_from_own_time_checkbox = wx.CheckBox(panel, -1, "From melody")
        self.melody_from_own_time_checkbox.SetValue(self.params.get("melody_from_own_time", True))
        self.melody_from_own_time_checkbox.Bind(wx.EVT_CHECKBOX, self.OnMelodyFromOwnTime)
        self.fullscreen_checkbox = wx.CheckBox(panel, -1, "Fullscreen network window (green/F11)")
        self.fullscreen_checkbox.SetValue(bool(self.params.get("fullscreen", False)))
        self.fullscreen_checkbox.Bind(wx.EVT_CHECKBOX, self.OnFullscreen)
        self.logic_worker_status_label = wx.StaticText(panel, -1, "Logic worker: idle")

        self.selection_exploration_label = wx.StaticText(panel, -1, "Selection exploration (0-1)")
        self.selection_exploration_ctrl = wx.TextCtrl(panel, -1, str(self.params["selection_exploration"]), style=wx.TE_PROCESS_ENTER)
        self._bindNumericCtrl(self.selection_exploration_ctrl, self.OnSelectionExploration)

        self.selection_top_k_label = wx.StaticText(panel, -1, "Selection top-k")
        self.selection_top_k_ctrl = wx.TextCtrl(panel, -1, str(self.params["selection_top_k"]), style=wx.TE_PROCESS_ENTER)
        self._bindNumericCtrl(self.selection_top_k_ctrl, self.OnSelectionTopK)

        self.jump_probability_label = wx.StaticText(panel, -1, "Jump probability (0-1)")
        self.jump_probability_ctrl = wx.TextCtrl(panel, -1, str(self.params["jump_probability"]), style=wx.TE_PROCESS_ENTER)
        self._bindNumericCtrl(self.jump_probability_ctrl, self.OnJumpProbability)

        self.jump_radius_label = wx.StaticText(panel, -1, "Jump radius (gematria)")
        self.jump_radius_ctrl = wx.TextCtrl(panel, -1, str(self.params["jump_radius"]), style=wx.TE_PROCESS_ENTER)
        self._bindNumericCtrl(self.jump_radius_ctrl, self.OnJumpRadius)

        self.use_phoneme_rhyme_checkbox = wx.CheckBox(panel, -1, "Use phoneme rhyme")
        self.use_phoneme_rhyme_checkbox.SetValue(self.params["use_phoneme_rhyme"])
        self.use_phoneme_rhyme_checkbox.Bind(wx.EVT_CHECKBOX, self.OnUsePhonemeRhyme)

        self.rhyme_weight_label = wx.StaticText(panel, -1, "Rhyme weight (0-1)")
        self.rhyme_weight_ctrl = wx.TextCtrl(panel, -1, str(self.params["rhyme_weight"]), style=wx.TE_PROCESS_ENTER)
        self._bindNumericCtrl(self.rhyme_weight_ctrl, self.OnRhymeWeight)

        self.rhyme_min_similarity_label = wx.StaticText(panel, -1, "Rhyme min similarity (0-1)")
        self.rhyme_min_similarity_ctrl = wx.TextCtrl(panel, -1, str(self.params["rhyme_min_similarity"]), style=wx.TE_PROCESS_ENTER)
        self._bindNumericCtrl(self.rhyme_min_similarity_ctrl, self.OnRhymeMinSimilarity)

        self.rhyme_strategy_label = wx.StaticText(panel, -1, "Rhyme strategy")
        self.rhyme_strategy_choice = wx.Choice(panel, -1, choices=[label for _, label in self._rhyme_strategy_modes()])
        self.rhyme_strategy_choice.SetStringSelection(self._rhyme_strategy_label_from_key(self.params.get("rhyme_strategy", "hybrid")))
        self.rhyme_strategy_choice.Bind(wx.EVT_CHOICE, self.OnRhymeStrategy)

        self.rhyme_tail_bias_label = wx.StaticText(panel, -1, "Rhyme tail bias (0-1)")
        self.rhyme_tail_bias_ctrl = wx.TextCtrl(panel, -1, str(self.params.get("rhyme_tail_bias", 0.60)), style=wx.TE_PROCESS_ENTER)
        self._bindNumericCtrl(self.rhyme_tail_bias_ctrl, self.OnRhymeTailBias)

        self.audio_wave_mode_label = wx.StaticText(panel, -1, "Audio waveform mode")
        self.audio_wave_mode_choice = wx.Choice(panel, -1, choices=[
            "Dynamic", "Pure sine", "Noise-heavy", "Classic analog",
            "Neuro formant", "Neuro pulse", "Neuro ring", "Neuro fold", "Neuro FM",
        ])
        self.audio_wave_mode_choice.SetSelection(0)
        self.audio_wave_mode_choice.Bind(wx.EVT_CHOICE, self.OnAudioWaveMode)

        self.piper_tts_checkbox = wx.CheckBox(panel, -1, "Piper TTS mode")
        self.piper_tts_checkbox.SetValue(bool(self.params.get("piper_tts_on", False)))
        self.piper_tts_checkbox.Bind(wx.EVT_CHECKBOX, self.OnPiperTTS)
        self.piper_model_label = wx.StaticText(panel, -1, "Piper model (.onnx)")
        self.piper_model_ctrl = wx.TextCtrl(panel, -1, str(self.params.get("piper_model_path", "")), style=wx.TE_PROCESS_ENTER)
        self.piper_model_ctrl.Bind(wx.EVT_TEXT_ENTER, self.OnPiperModelPath)
        self.piper_model_browse_button = wx.Button(panel, -1, "Browse...")
        self.piper_model_browse_button.Bind(wx.EVT_BUTTON, self.OnPiperModelBrowse)
        self.piper_tts_status_label = wx.StaticText(panel, -1, "Piper TTS: idle")

        self.synth_volume_label = wx.StaticText(panel, -1, "Synth volume: 100%")
        _sv_init = int(self.params.get("synth_volume", 1.0) * 100)
        self.synth_volume_slider = wx.Slider(panel, -1, _sv_init, 0, 100, style=wx.SL_HORIZONTAL)
        self.synth_volume_slider.Bind(wx.EVT_SLIDER, self.OnSynthVolume)

        self.compressor_enabled_checkbox = wx.CheckBox(panel, -1, "Enable synth compressor")
        self.compressor_enabled_checkbox.SetValue(bool(self.params.get("compressor_enabled", False)))
        self.compressor_enabled_checkbox.Bind(wx.EVT_CHECKBOX, self.OnCompressorEnabled)
        _ct_init = int(round(float(self.params.get("compressor_threshold_db", -18.0))))
        _ct_init = min(0, max(-48, _ct_init))
        self.compressor_threshold_label = wx.StaticText(panel, -1, "Compressor threshold: -18 dB")
        self.compressor_threshold_slider = wx.Slider(panel, -1, _ct_init, -48, 0, style=wx.SL_HORIZONTAL)
        self.compressor_threshold_slider.Bind(wx.EVT_SLIDER, self.OnCompressorThreshold)
        _cr_init = int(round(float(self.params.get("compressor_ratio", 3.0)) * 10.0))
        _cr_init = min(120, max(10, _cr_init))
        self.compressor_ratio_label = wx.StaticText(panel, -1, "Compressor ratio: 3.0:1")
        self.compressor_ratio_slider = wx.Slider(panel, -1, _cr_init, 10, 120, style=wx.SL_HORIZONTAL)
        self.compressor_ratio_slider.Bind(wx.EVT_SLIDER, self.OnCompressorRatio)
        _cm_init = int(round(float(self.params.get("compressor_makeup_db", 6.0))))
        _cm_init = min(24, max(0, _cm_init))
        self.compressor_makeup_label = wx.StaticText(panel, -1, "Compressor makeup gain: +6 dB")
        self.compressor_makeup_slider = wx.Slider(panel, -1, _cm_init, 0, 24, style=wx.SL_HORIZONTAL)
        self.compressor_makeup_slider.Bind(wx.EVT_SLIDER, self.OnCompressorMakeup)

        _pv_pct = int(self.params.get("piper_volume", 0.5) * 100)
        self.piper_volume_label = wx.StaticText(panel, -1, f"Piper volume: {_pv_pct}%")
        self.piper_volume_slider = wx.Slider(panel, -1, _pv_pct, 0, 100, style=wx.SL_HORIZONTAL)
        self.piper_volume_slider.Bind(wx.EVT_SLIDER, self.OnPiperVolume)

        self.frequency_mapping_label = wx.StaticText(panel, -1, "Frequency mapping")
        self.frequency_mapping_choice = wx.Choice(panel, -1, choices=[label for _, label in self._frequency_mapping_modes()])
        self.frequency_mapping_choice.SetStringSelection(self._frequency_mapping_label_from_key(self.params.get("frequency_mapping_mode", "original_notes")))
        self.frequency_mapping_choice.Bind(wx.EVT_CHOICE, self.OnFrequencyMappingMode)

        self.voice_count_label = wx.StaticText(panel, -1, "Polyphony voices")
        self.voice_count_choice = wx.Choice(panel, -1, choices=["1", "2", "3", "4"])
        self.voice_count_choice.SetSelection(0)
        self.voice_count_choice.Bind(wx.EVT_CHOICE, self.OnVoiceCount)

        self.rhythm_style_label = wx.StaticText(panel, -1, "Rhythm style")
        self.rhythm_style_choice = wx.Choice(panel, -1, choices=[label for _, label in self._rhythm_style_modes()])
        self.rhythm_style_choice.SetStringSelection(self._rhythm_style_label_from_key(self.params.get("rhythm_style", "manual")))
        self.rhythm_style_choice.Bind(wx.EVT_CHOICE, self.OnRhythmStyle)

        self.beat_library_style_label = wx.StaticText(panel, -1, "Beat library")
        self.beat_library_style_choice = wx.Choice(panel, -1, choices=[label for _, label in self._beat_library_modes()])
        self.beat_library_style_choice.SetStringSelection(self._beat_library_label_from_key(self.params.get("beat_library_style", "auto")))
        self.beat_library_style_choice.Bind(wx.EVT_CHOICE, self.OnBeatLibraryStyle)

        self.strict_counterpoint_checkbox = wx.CheckBox(panel, -1, "Strict CP")
        self.strict_counterpoint_checkbox.SetValue(self.params["strict_counterpoint"])
        self.strict_counterpoint_checkbox.Bind(wx.EVT_CHECKBOX, self.OnStrictCounterpoint)

        self.voice_spread_label = wx.StaticText(panel, -1, "Voice spread")
        self.voice_spread_ctrl = wx.TextCtrl(panel, -1, str(self.params["voice_spread"]), style=wx.TE_PROCESS_ENTER)
        self._bindNumericCtrl(self.voice_spread_ctrl, self.OnVoiceSpread)

        self.voice_distance_label = wx.StaticText(panel, -1, "Voice distance (0-1)")
        self.voice_distance_ctrl = wx.TextCtrl(panel, -1, str(self.params["voice_distance"]), style=wx.TE_PROCESS_ENTER)
        self._bindNumericCtrl(self.voice_distance_ctrl, self.OnVoiceDistance)

        self.voice_distance_context_label = wx.StaticText(panel, -1, "Voice distance context")
        self.voice_distance_context_ctrl = wx.TextCtrl(panel, -1, str(self.params["voice_distance_context"]), style=wx.TE_PROCESS_ENTER)
        self._bindNumericCtrl(self.voice_distance_context_ctrl, self.OnVoiceDistanceContext)

        self.rhythmic_divergence_label = wx.StaticText(panel, -1, "Rhythmic divergence (0-1)")
        self.rhythmic_divergence_ctrl = wx.TextCtrl(panel, -1, str(self.params["rhythmic_divergence"]), style=wx.TE_PROCESS_ENTER)
        self._bindNumericCtrl(self.rhythmic_divergence_ctrl, self.OnRhythmicDivergence)

        self.rhythm_gate_strength_label = wx.StaticText(panel, -1, "Rhythm gate strength (0-1)")
        self.rhythm_gate_strength_ctrl = wx.TextCtrl(panel, -1, str(self.params["rhythm_gate_strength"]), style=wx.TE_PROCESS_ENTER)
        self._bindNumericCtrl(self.rhythm_gate_strength_ctrl, self.OnRhythmGateStrength)

        self.rhythm_stretch_strength_label = wx.StaticText(panel, -1, "Rhythm stretch strength (0-1)")
        self.rhythm_stretch_strength_ctrl = wx.TextCtrl(panel, -1, str(self.params["rhythm_stretch_strength"]), style=wx.TE_PROCESS_ENTER)
        self._bindNumericCtrl(self.rhythm_stretch_strength_ctrl, self.OnRhythmStretchStrength)

        self.rhythm_rotation_label = wx.StaticText(panel, -1, "Rhythm rotation")
        self.rhythm_rotation_ctrl = wx.TextCtrl(panel, -1, str(self.params["rhythm_rotation"]), style=wx.TE_PROCESS_ENTER)
        self._bindNumericCtrl(self.rhythm_rotation_ctrl, self.OnRhythmRotation)

        self.rhythm_radicality_label = wx.StaticText(panel, -1, "Rhythm radicality (0-1)")
        self.rhythm_radicality_ctrl = wx.TextCtrl(panel, -1, str(self.params["rhythm_radicality"]), style=wx.TE_PROCESS_ENTER)
        self._bindNumericCtrl(self.rhythm_radicality_ctrl, self.OnRhythmRadicality)

        self.rhythm_mod_bpm_label = wx.StaticText(panel, -1, "Rhythm modulation BPM")
        self.rhythm_mod_bpm_ctrl = wx.TextCtrl(panel, -1, str(self.params.get("rhythm_mod_bpm", 108.0)), style=wx.TE_PROCESS_ENTER)
        self._bindNumericCtrl(self.rhythm_mod_bpm_ctrl, self.OnRhythmModBPM)

        self.additive_editor_button = wx.Button(panel, -1, "Edit additive rhythm blocks")
        self.additive_editor_button.Bind(wx.EVT_BUTTON, self.OnOpenAdditiveRhythmEditor)
        self.additive_pattern_status = wx.StaticText(panel, -1, "")

        self.divisive_signature_label = wx.StaticText(panel, -1, "Divisive time signature")
        self.divisive_signature_choice = wx.Choice(panel, -1, choices=[label for _, label in self._divisive_signature_modes()])
        self.divisive_signature_choice.SetStringSelection(self._divisive_signature_label_from_key(self.params.get("divisive_rhythm_signature", "4/4")))
        self.divisive_signature_choice.Bind(wx.EVT_CHOICE, self.OnDivisiveRhythmSignature)

        _aw_pct = int(self.params.get("additive_rhythm_weight", 0.0) * 100)
        self.additive_rhythm_weight_label = wx.StaticText(panel, -1, f"Additive rhythm weight: {_aw_pct}%")
        self.additive_rhythm_weight_slider = wx.Slider(panel, -1, _aw_pct, 0, 100, style=wx.SL_HORIZONTAL)
        self.additive_rhythm_weight_slider.Bind(wx.EVT_SLIDER, self.OnAdditiveRhythmWeight)

        _dw_pct = int(self.params.get("divisive_rhythm_weight", 0.0) * 100)
        self.divisive_rhythm_weight_label = wx.StaticText(panel, -1, f"Divisive rhythm weight: {_dw_pct}%")
        self.divisive_rhythm_weight_slider = wx.Slider(panel, -1, _dw_pct, 0, 100, style=wx.SL_HORIZONTAL)
        self.divisive_rhythm_weight_slider.Bind(wx.EVT_SLIDER, self.OnDivisiveRhythmWeight)

        self.melody_coherence_label = wx.StaticText(panel, -1, "Melody coherence (0-1)")
        self.melody_coherence_ctrl = wx.TextCtrl(panel, -1, str(self.params["melody_coherence"]), style=wx.TE_PROCESS_ENTER)
        self._bindNumericCtrl(self.melody_coherence_ctrl, self.OnMelodyCoherence)

        self.melody_speed_label = wx.StaticText(panel, -1, "Melody speed coeff")
        self.melody_speed_ctrl = wx.TextCtrl(panel, -1, str(self.params["melody_speed"]), style=wx.TE_PROCESS_ENTER)
        self._bindNumericCtrl(self.melody_speed_ctrl, self.OnMelodySpeed)

        self.min_note_duration_label = wx.StaticText(panel, -1, "Minimum note duration (s)")
        self.min_note_duration_ctrl = wx.TextCtrl(panel, -1, str(self.params["min_note_duration"]), style=wx.TE_PROCESS_ENTER)
        self._bindNumericCtrl(self.min_note_duration_ctrl, self.OnMinNoteDuration)

        self.adsr_label = wx.StaticText(panel, -1, "ADSR envelope")
        self.adsr_display = ADSRDisplayPanel(panel)
        self.adsr_attack_label = wx.StaticText(panel, -1, "Attack (s)")
        self.adsr_attack_ctrl = wx.TextCtrl(panel, -1, str(self.params["adsr_attack"]), style=wx.TE_PROCESS_ENTER)
        self._bindNumericCtrl(self.adsr_attack_ctrl, self.OnADSRAttack)
        self.adsr_decay_label = wx.StaticText(panel, -1, "Decay (s)")
        self.adsr_decay_ctrl = wx.TextCtrl(panel, -1, str(self.params["adsr_decay"]), style=wx.TE_PROCESS_ENTER)
        self._bindNumericCtrl(self.adsr_decay_ctrl, self.OnADSRDecay)
        self.adsr_sustain_label = wx.StaticText(panel, -1, "Sustain (0-1)")
        self.adsr_sustain_ctrl = wx.TextCtrl(panel, -1, str(self.params["adsr_sustain"]), style=wx.TE_PROCESS_ENTER)
        self._bindNumericCtrl(self.adsr_sustain_ctrl, self.OnADSRSustain)
        self.adsr_release_label = wx.StaticText(panel, -1, "Release (s)")
        self.adsr_release_ctrl = wx.TextCtrl(panel, -1, str(self.params["adsr_release"]), style=wx.TE_PROCESS_ENTER)
        self._bindNumericCtrl(self.adsr_release_ctrl, self.OnADSRRelease)

        self.add_words_label = wx.StaticText(panel, -1, "Add word(s)")
        self.add_words_ctrl = wx.TextCtrl(panel, -1, "", style=wx.TE_PROCESS_ENTER)
        self.add_words_ctrl.Bind(wx.EVT_TEXT_ENTER, self.OnAddWords)
        self.add_words_button = wx.Button(panel, -1, "Add")
        self.add_words_button.Bind(wx.EVT_BUTTON, self.OnAddWords)
        self.clear_sentence_button = wx.Button(panel, -1, "Clear sentence")
        self.clear_sentence_button.Bind(wx.EVT_BUTTON, self.OnClearSentence)
        self.add_words_status = wx.StaticText(panel, -1, "")

        self.import_db_label = wx.StaticText(panel, -1, "Import database file")
        self.import_mode_label = wx.StaticText(panel, -1, "Import mode")
        self.import_mode_choice = wx.Choice(panel, -1, choices=["Append database", "Replace database"])
        self.import_mode_choice.SetSelection(0)
        self.import_mode_choice.Bind(wx.EVT_CHOICE, self.OnImportMode)
        self.import_db_button = wx.Button(panel, -1, "Import .txt")
        self.import_db_button.Bind(wx.EVT_BUTTON, self.OnImportDatabaseFile)
        self.import_db_status = wx.StaticText(panel, -1, "")

        self.preset_label = wx.StaticText(panel, -1, "Presets (.json)")
        self.preset_save_button = wx.Button(panel, -1, "Save preset")
        self.preset_save_button.Bind(wx.EVT_BUTTON, self.OnSavePreset)
        self.preset_load_button = wx.Button(panel, -1, "Load preset")
        self.preset_load_button.Bind(wx.EVT_BUTTON, self.OnLoadPreset)
        self.preset_status = wx.StaticText(panel, -1, "")

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.set_weight_by_gematria_checkbox, 0, wx.ALL, 5)
        self.sizer.Add(self.use_pos_matching_checkbox, 0, wx.ALL, 5)
        self.sizer.Add(self.pos_backend_status_label, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)
        self.sizer.Add(self.fluid_pos_checkbox, 0, wx.ALL, 5)
        self.sizer.Add(self.use_long_term_memory_checkbox, 0, wx.ALL, 5)
        self.sizer.Add(self.ltm_weight_label, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.sizer.Add(self.ltm_weight_ctrl, 0, wx.ALL, 5)
        self.sizer.Add(self.common_word_penalty_checkbox, 0, wx.ALL, 5)
        self.sizer.Add(self.ltm_load_button, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.sizer.Add(self.ltm_status_label, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)
        self.sizer.Add(self.fluid_root_checkbox, 0, wx.ALL, 5)
        self.sizer.Add(self.fluid_gematria_checkbox, 0, wx.ALL, 5)

        self.sizer.Add(self.learning_rate_label, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.sizer.Add(self.learning_rate_ctrl, 0, wx.ALL, 5)

        self.sizer.Add(self.error_label, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.sizer.Add(self.error_ctrl, 0, wx.ALL, 5)

        self.sizer.Add(self.activation_increase_label, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.sizer.Add(self.activation_increase_ctrl, 0, wx.ALL, 5)

        self.sizer.Add(self.activation_limit_label, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.sizer.Add(self.activation_limit_ctrl, 0, wx.ALL, 5)

        self.sizer.Add(self.sigmoid_scale_label, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.sizer.Add(self.sigmoid_scale_ctrl, 0, wx.ALL, 5)

        self.sizer.Add(self.word_change_threshold_label, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.sizer.Add(self.word_change_threshold_ctrl, 0, wx.ALL, 5)

        self.sizer.Add(self.zoom_label, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.sizer.Add(self.zoom_ctrl, 0, wx.ALL, 5)

        self.sizer.Add(self.process_interval_label, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        _pi_row = wx.BoxSizer(wx.HORIZONTAL)
        _pi_row.Add(self.process_interval_ctrl, 0, wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 8)
        _pi_row.Add(self.melody_from_own_time_checkbox, 0, wx.ALIGN_CENTER_VERTICAL)
        _pi_row.Add(self.process_interval_bpm_sync_checkbox, 0, wx.LEFT | wx.ALIGN_CENTER_VERTICAL, 10)
        _pi_row.Add(self.process_interval_value_label, 0, wx.LEFT | wx.ALIGN_CENTER_VERTICAL, 10)
        self.sizer.Add(_pi_row, 0, wx.ALL, 5)
        self.sizer.Add(self.fullscreen_checkbox, 0, wx.ALL, 5)
        self.sizer.Add(self.logic_worker_status_label, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)

        self.sizer.Add(self.selection_exploration_label, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.sizer.Add(self.selection_exploration_ctrl, 0, wx.ALL, 5)

        self.sizer.Add(self.selection_top_k_label, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.sizer.Add(self.selection_top_k_ctrl, 0, wx.ALL, 5)

        self.sizer.Add(self.jump_probability_label, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.sizer.Add(self.jump_probability_ctrl, 0, wx.ALL, 5)

        self.sizer.Add(self.jump_radius_label, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.sizer.Add(self.jump_radius_ctrl, 0, wx.ALL, 5)

        self.sizer.Add(self.use_phoneme_rhyme_checkbox, 0, wx.ALL, 5)
        self.sizer.Add(self.rhyme_weight_label, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.sizer.Add(self.rhyme_weight_ctrl, 0, wx.ALL, 5)
        self.sizer.Add(self.rhyme_min_similarity_label, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.sizer.Add(self.rhyme_min_similarity_ctrl, 0, wx.ALL, 5)
        self.sizer.Add(self.rhyme_strategy_label, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.sizer.Add(self.rhyme_strategy_choice, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.sizer.Add(self.rhyme_tail_bias_label, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.sizer.Add(self.rhyme_tail_bias_ctrl, 0, wx.ALL, 5)

        self.sizer.Add(self.audio_wave_mode_label, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.sizer.Add(self.audio_wave_mode_choice, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.sizer.Add(self.piper_tts_checkbox, 0, wx.ALL, 5)
        self.sizer.Add(self.piper_model_label, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        _piper_model_row = wx.BoxSizer(wx.HORIZONTAL)
        _piper_model_row.Add(self.piper_model_ctrl, 1, wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 5)
        _piper_model_row.Add(self.piper_model_browse_button, 0, wx.ALIGN_CENTER_VERTICAL)
        self.sizer.Add(_piper_model_row, 0, wx.EXPAND | wx.ALL, 5)
        self.sizer.Add(self.piper_tts_status_label, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.sizer.Add(self.synth_volume_label, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.sizer.Add(self.synth_volume_slider, 0, wx.EXPAND | wx.ALL, 5)
        self.sizer.Add(self.compressor_enabled_checkbox, 0, wx.ALL, 5)
        self.sizer.Add(self.compressor_threshold_label, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.sizer.Add(self.compressor_threshold_slider, 0, wx.EXPAND | wx.ALL, 5)
        self.sizer.Add(self.compressor_ratio_label, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.sizer.Add(self.compressor_ratio_slider, 0, wx.EXPAND | wx.ALL, 5)
        self.sizer.Add(self.compressor_makeup_label, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.sizer.Add(self.compressor_makeup_slider, 0, wx.EXPAND | wx.ALL, 5)
        self.sizer.Add(self.piper_volume_label, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.sizer.Add(self.piper_volume_slider, 0, wx.EXPAND | wx.ALL, 5)
        self.sizer.Add(self.frequency_mapping_label, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.sizer.Add(self.frequency_mapping_choice, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.sizer.Add(self.rhythm_style_label, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.sizer.Add(self.rhythm_style_choice, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.sizer.Add(self.beat_library_style_label, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.sizer.Add(self.beat_library_style_choice, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.sizer.Add(self.voice_count_label, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.sizer.Add(self.voice_count_choice, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.sizer.Add(self.strict_counterpoint_checkbox, 0, wx.ALL, 5)
        self.sizer.Add(self.voice_spread_label, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.sizer.Add(self.voice_spread_ctrl, 0, wx.ALL, 5)
        self.sizer.Add(self.voice_distance_label, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.sizer.Add(self.voice_distance_ctrl, 0, wx.ALL, 5)
        self.sizer.Add(self.voice_distance_context_label, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.sizer.Add(self.voice_distance_context_ctrl, 0, wx.ALL, 5)
        self.sizer.Add(self.rhythmic_divergence_label, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.sizer.Add(self.rhythmic_divergence_ctrl, 0, wx.ALL, 5)
        self.sizer.Add(self.rhythm_gate_strength_label, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.sizer.Add(self.rhythm_gate_strength_ctrl, 0, wx.ALL, 5)
        self.sizer.Add(self.rhythm_stretch_strength_label, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.sizer.Add(self.rhythm_stretch_strength_ctrl, 0, wx.ALL, 5)
        self.sizer.Add(self.rhythm_rotation_label, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.sizer.Add(self.rhythm_rotation_ctrl, 0, wx.ALL, 5)
        self.sizer.Add(self.rhythm_radicality_label, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.sizer.Add(self.rhythm_radicality_ctrl, 0, wx.ALL, 5)
        self.sizer.Add(self.rhythm_mod_bpm_label, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.sizer.Add(self.rhythm_mod_bpm_ctrl, 0, wx.ALL, 5)
        self.sizer.Add(self.additive_editor_button, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.sizer.Add(self.additive_pattern_status, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)
        self.sizer.Add(self.divisive_signature_label, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.sizer.Add(self.divisive_signature_choice, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.sizer.Add(self.additive_rhythm_weight_label, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.sizer.Add(self.additive_rhythm_weight_slider, 0, wx.EXPAND | wx.ALL, 5)
        self.sizer.Add(self.divisive_rhythm_weight_label, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.sizer.Add(self.divisive_rhythm_weight_slider, 0, wx.EXPAND | wx.ALL, 5)
        self.sizer.Add(self.melody_coherence_label, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.sizer.Add(self.melody_coherence_ctrl, 0, wx.ALL, 5)
        self.sizer.Add(self.melody_speed_label, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.sizer.Add(self.melody_speed_ctrl, 0, wx.ALL, 5)
        self.sizer.Add(self.min_note_duration_label, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.sizer.Add(self.min_note_duration_ctrl, 0, wx.ALL, 5)

        self.sizer.Add(self.adsr_label, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.sizer.Add(self.adsr_display, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 5)

        adsr_grid = wx.FlexGridSizer(2, 4, 4, 5)
        adsr_grid.Add(self.adsr_attack_label, 0, wx.ALIGN_LEFT)
        adsr_grid.Add(self.adsr_decay_label, 0, wx.ALIGN_LEFT)
        adsr_grid.Add(self.adsr_sustain_label, 0, wx.ALIGN_LEFT)
        adsr_grid.Add(self.adsr_release_label, 0, wx.ALIGN_LEFT)
        adsr_grid.Add(self.adsr_attack_ctrl, 0, wx.EXPAND)
        adsr_grid.Add(self.adsr_decay_ctrl, 0, wx.EXPAND)
        adsr_grid.Add(self.adsr_sustain_ctrl, 0, wx.EXPAND)
        adsr_grid.Add(self.adsr_release_ctrl, 0, wx.EXPAND)
        self.sizer.Add(adsr_grid, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 5)

        add_words_row = wx.BoxSizer(wx.HORIZONTAL)
        add_words_row.Add(self.add_words_ctrl, 1, wx.RIGHT, 5)
        add_words_row.Add(self.add_words_button, 0)
        add_words_row.Add(self.clear_sentence_button, 0, wx.LEFT, 5)
        self.sizer.Add(self.add_words_label, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.sizer.Add(add_words_row, 0, wx.EXPAND | wx.ALL, 5)
        self.sizer.Add(self.add_words_status, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)

        self.sizer.Add(self.import_db_label, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.sizer.Add(self.import_mode_label, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.sizer.Add(self.import_mode_choice, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.sizer.Add(self.import_db_button, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.sizer.Add(self.import_db_status, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)

        preset_row = wx.BoxSizer(wx.HORIZONTAL)
        preset_row.Add(self.preset_save_button, 0, wx.RIGHT, 5)
        preset_row.Add(self.preset_load_button, 0)
        self.sizer.Add(self.preset_label, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.sizer.Add(preset_row, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.sizer.Add(self.preset_status, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)

        self._applyMonospaceToWindow(panel)
        panel.SetSizer(self.sizer)
        panel.FitInside()
        self.sizer.Fit(self.frame)

        best_width, best_height = self.frame.GetBestSize()
        display_width, display_height = wx.GetDisplaySize()
        target_width = min(max(760, best_width + 20), max(500, display_width - 80))
        target_height = min(max(900, best_height + 20), max(500, display_height - 80))

        self.frame.SetSize(target_width, target_height)
        self.frame.SetMinSize((420, 360))
        self.frame.Layout()
        self.app.SetTopWindow(self.frame)
        self.frame.Show()
        self.setupOutputWindow()
        self._applyADSRToAudio()
        self._updatePOSBackendStatusLabel(check_nltk=False)
        self._updateLTMStatusLabel()
        self._update_additive_pattern_status_label()
        self._sync_process_interval_from_bpm(force=True)
        self._update_process_interval_mode_ui()
        self._apply_rhythm_modulation_state()
        self._update_compressor_labels()
        self._apply_compressor_state()

    def setupOutputWindow(self):
        self.output_frame = wx.Frame(None, -1, "SanaVerkko Output", size=(760, 420))
        output_panel = wx.Panel(self.output_frame, -1)
        output_sizer = wx.BoxSizer(wx.VERTICAL)
        self.output_text_ctrl = wx.TextCtrl(
            output_panel,
            -1,
            "",
            style=wx.TE_MULTILINE | wx.TE_READONLY | wx.HSCROLL,
        )
        self.output_text_ctrl.SetFont(
            wx.Font(
                11,
                wx.FONTFAMILY_TELETYPE,
                wx.FONTSTYLE_NORMAL,
                wx.FONTWEIGHT_NORMAL,
                False,
                "Menlo",
            )
        )
        output_sizer.Add(self.output_text_ctrl, 1, wx.EXPAND | wx.ALL, 6)
        output_panel.SetSizer(output_sizer)

        frame_x, frame_y = self.frame.GetPosition()
        frame_w, _ = self.frame.GetSize()
        self.output_frame.SetPosition((frame_x + frame_w + 10, frame_y))
        self.output_frame.Bind(wx.EVT_CLOSE, self.OnOutputFrameClose)
        self.output_frame.Show()

        self.output_timer = wx.Timer(self.output_frame)
        self.output_frame.Bind(wx.EVT_TIMER, self.OnOutputTimer, self.output_timer)
        self.output_timer.Start(300)
        self.refreshOutputWindow()

    def OnOutputTimer(self, event):
        self.refreshOutputWindow()

    def OnOutputFrameClose(self, event):
        if self.output_timer is not None:
            self.output_timer.Stop()
            self.output_timer = None
        self.output_frame = None
        self.output_text_ctrl = None
        event.Skip()

    def refreshOutputWindow(self):
        if self.output_text_ctrl is None:
            return

        try:
            with open(self.output_file_path, "r") as infile:
                content = infile.read()
        except Exception:
            content = ""

        if content == self.last_output_content:
            return

        self.last_output_content = content
        self.output_text_ctrl.SetValue(content)
        self.output_text_ctrl.ShowPosition(self.output_text_ctrl.GetLastPosition())

    def OnSetWeightByGematria(self, event):
        if self._suppress_param_events:
            return
        self.params["set_weight_by_gematria"] = self.set_weight_by_gematria_checkbox.GetValue()

    def OnUsePOSMatching(self, event):
        if self._suppress_param_events:
            return
        self.params["use_pos_matching"] = self._is_pos_matching_enabled()
        self._updatePOSBackendStatusLabel(check_nltk=self.params["use_pos_matching"])

    def OnFluidPOS(self, event):
        if self._suppress_param_events:
            return
        self.params["fluid_pos"] = self._is_fluid_pos_enabled()

    def OnUseLongTermMemory(self, event):
        if self._suppress_param_events:
            return
        self.params["use_long_term_memory"] = self._is_ltm_enabled()
        self._updateLTMStatusLabel()

    def OnLTMWeight(self, event):
        self._commit_float_param(self.ltm_weight_ctrl, "ltm_weight", minimum=0.0, maximum=1.0)

    def OnCommonWordPenalty(self, event):
        if self._suppress_param_events:
            return
        self.params["common_word_penalty"] = bool(self.common_word_penalty_checkbox.GetValue())

    def OnLoadLongTermMemory(self, event):
        with wx.FileDialog(
            self.frame,
            "Load long term memory model",
            wildcard="LTM model (*.svltm)|*.svltm|All files (*.*)|*.*",
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
        ) as file_dialog:
            if file_dialog.ShowModal() == wx.ID_CANCEL:
                return
            selected_path = file_dialog.GetPath()

        try:
            self._load_ltm_model(selected_path)
            self._updateLTMStatusLabel()
        except Exception as error:
            self.ltm_status_label.SetLabel(f"LTM load failed: {error}")

    def OnFluidRoot(self, event):
        if self._suppress_param_events:
            return
        self.params["fluid_root"] = self._is_fluid_root_enabled()

    def OnFluidGematria(self, event):
        if self._suppress_param_events:
            return
        self.params["fluid_gematria"] = self._is_fluid_gematria_enabled()

    def _is_pos_matching_enabled(self):
        try:
            if hasattr(self, "use_pos_matching_checkbox") and self.use_pos_matching_checkbox is not None:
                return bool(self.use_pos_matching_checkbox.GetValue())
        except Exception:
            pass
        return bool(self.params.get("use_pos_matching", False))

    def _is_fluid_root_enabled(self):
        try:
            if hasattr(self, "fluid_root_checkbox") and self.fluid_root_checkbox is not None:
                return bool(self.fluid_root_checkbox.GetValue())
        except Exception:
            pass
        return bool(self.params.get("fluid_root", False))

    def _is_fluid_gematria_enabled(self):
        try:
            if hasattr(self, "fluid_gematria_checkbox") and self.fluid_gematria_checkbox is not None:
                return bool(self.fluid_gematria_checkbox.GetValue())
        except Exception:
            pass
        return bool(self.params.get("fluid_gematria", False))

    def _is_fluid_pos_enabled(self):
        try:
            if hasattr(self, "fluid_pos_checkbox") and self.fluid_pos_checkbox is not None:
                return bool(self.fluid_pos_checkbox.GetValue())
        except Exception:
            pass
        return bool(self.params.get("fluid_pos", False))

    def _is_ltm_enabled(self):
        try:
            if hasattr(self, "use_long_term_memory_checkbox") and self.use_long_term_memory_checkbox is not None:
                return bool(self.use_long_term_memory_checkbox.GetValue())
        except Exception:
            pass
        return bool(self.params.get("use_long_term_memory", False))

    def _load_ltm_model(self, model_path):
        if sv_ltm is None:
            raise RuntimeError("sv_ltm module is not available")
        self.ltm_model = sv_ltm.load_model(model_path)
        self.ltm_model_path = model_path
        self.ltm_context_size = max(1, int(getattr(self.ltm_model, "context_size", 3)))

    def _updateLTMStatusLabel(self):
        if not hasattr(self, "ltm_status_label") or self.ltm_status_label is None:
            return

        if sv_ltm is None:
            self.ltm_status_label.SetLabel("LTM: module unavailable")
            return

        enabled = self._is_ltm_enabled()
        if self.ltm_model is None:
            if enabled:
                self.ltm_status_label.SetLabel("LTM: enabled (no model loaded)")
            else:
                self.ltm_status_label.SetLabel("LTM: disabled")
            return

        model_name = os.path.basename(self.ltm_model_path) if self.ltm_model_path else "model loaded"
        backend_name = "cpu"
        try:
            if hasattr(self.ltm_model, "runtime_backend"):
                backend_name = str(self.ltm_model.runtime_backend())
        except Exception:
            backend_name = "cpu"
        if enabled:
            self.ltm_status_label.SetLabel(f"LTM: enabled ({model_name}, backend={backend_name})")
        else:
            self.ltm_status_label.SetLabel(f"LTM: loaded ({model_name}, backend={backend_name}), disabled")

    def _ensure_nltk_pos_tagger(self):
        if nltk is None:
            self._updatePOSBackendStatusLabel(check_nltk=False)
            return False

        if self.nltk_pos_ready:
            self._updatePOSBackendStatusLabel(check_nltk=False)
            return True

        if self.nltk_pos_init_attempted:
            self._updatePOSBackendStatusLabel(check_nltk=False)
            return False

        self.nltk_pos_init_attempted = True
        try:
            nltk.pos_tag(["word"])
            self.nltk_pos_ready = True
            self._updatePOSBackendStatusLabel(check_nltk=False)
            return True
        except LookupError:
            for resource in ["averaged_perceptron_tagger_eng", "averaged_perceptron_tagger"]:
                try:
                    nltk.download(resource, quiet=True)
                except Exception:
                    pass
            try:
                nltk.pos_tag(["word"])
                self.nltk_pos_ready = True
                self._updatePOSBackendStatusLabel(check_nltk=False)
                return True
            except Exception:
                self._updatePOSBackendStatusLabel(check_nltk=False)
                return False
        except Exception:
            self._updatePOSBackendStatusLabel(check_nltk=False)
            return False

    def _updatePOSBackendStatusLabel(self, check_nltk=False):
        if not hasattr(self, "pos_backend_status_label") or self.pos_backend_status_label is None:
            return

        pos_enabled = self._is_pos_matching_enabled()
        if not pos_enabled:
            self.pos_backend_status_label.SetLabel("POS backend: disabled")
            return

        if check_nltk:
            nltk_ready = self._ensure_nltk_pos_tagger()
        else:
            nltk_ready = bool(self.nltk_pos_ready)

        if nltk_ready:
            self.pos_backend_status_label.SetLabel("POS backend: NLTK")
        elif nltk is None:
            self.pos_backend_status_label.SetLabel("POS backend: heuristic (NLTK missing)")
        else:
            self.pos_backend_status_label.SetLabel("POS backend: heuristic (NLTK unavailable)")

    def _heuristic_pos_tag(self, word_text):
        text = word_text.lower()
        if text.endswith("ly"):
            return "RB"
        if text.endswith("ing") or text.endswith("ed"):
            return "VB"
        if text.endswith(("ous", "ful", "ive", "al", "ic", "ish", "less")):
            return "JJ"
        if text.endswith(("ness", "tion", "sion", "ment", "ity", "ism", "ship")):
            return "NN"
        return "NN"

    def getWordPOS(self, word_text, force=False):
        if not force and not self._is_pos_matching_enabled():
            return ""

        key = word_text.lower().strip()
        if key in self.pos_tag_cache:
            return self.pos_tag_cache[key]

        pos_tag = None
        if self._ensure_nltk_pos_tagger():
            try:
                pos_tag = nltk.pos_tag([key])[0][1]
            except Exception:
                pos_tag = None

        if pos_tag is None:
            pos_tag = self._heuristic_pos_tag(key)

        self.pos_tag_cache[key] = pos_tag
        return pos_tag

    def _assign_seed_pos(self, word_obj):
        if word_obj is None:
            return ""
        try:
            pos_value = self.getWordPOS(word_obj.word, force=True)
        except Exception:
            pos_value = ""
        word_obj.seed_pos = pos_value
        return pos_value

    def _get_seed_pos(self, word_obj):
        if word_obj is None:
            return ""
        pos_value = getattr(word_obj, "seed_pos", "")
        if isinstance(pos_value, str) and pos_value != "":
            return pos_value
        return self._assign_seed_pos(word_obj)

    def _bindNumericCtrl(self, ctrl, handler):
        ctrl.Unbind(wx.EVT_TEXT_ENTER, handler=handler)
        ctrl.Bind(wx.EVT_TEXT_ENTER, handler)

    def _setCtrlValueSilently(self, ctrl, value):
        if ctrl is None:
            return
        try:
            ctrl.ChangeValue(str(value))
        except Exception:
            ctrl.SetValue(str(value))

    def _getMonospaceWxFont(self, point_size=11):
        return wx.Font(
            point_size,
            wx.FONTFAMILY_TELETYPE,
            wx.FONTSTYLE_NORMAL,
            wx.FONTWEIGHT_NORMAL,
            False,
            "Menlo",
        )

    def _applyMonospaceToWindow(self, window, font=None):
        if window is None:
            return

        if font is None:
            font = self._getMonospaceWxFont(11)

        try:
            window.SetFont(font)
        except Exception:
            pass

        try:
            children = window.GetChildren()
        except Exception:
            children = []

        for child in children:
            self._applyMonospaceToWindow(child, font)

    def _readFloat(self, ctrl):
        text_value = ctrl.GetValue().strip().replace(",", ".")
        if text_value == "":
            return None
        try:
            return float(text_value)
        except ValueError:
            return None

    def _readInt(self, ctrl):
        text_value = ctrl.GetValue().strip()
        if text_value == "":
            return None
        try:
            return int(float(text_value))
        except ValueError:
            return None

    def _is_duplicate_param_event(self, event, fallback_ctrl=None):
        if self._suppress_param_events or self._in_param_commit:
            return True
        return False

    def _commit_float_param(self, ctrl, key, minimum=None, maximum=None):
        if self._is_duplicate_param_event(None, ctrl):
            return None
        value = self._readFloat(ctrl)
        if value is None:
            return None

        normalized = float(value)
        if minimum is not None:
            normalized = max(float(minimum), normalized)
        if maximum is not None:
            normalized = min(float(maximum), normalized)

        self._in_param_commit = True
        try:
            self.params[key] = normalized
            self._setCtrlValueSilently(ctrl, normalized)
        finally:
            self._in_param_commit = False
        return normalized

    def _commit_int_param(self, ctrl, key, minimum=None, maximum=None):
        if self._is_duplicate_param_event(None, ctrl):
            return None
        value = self._readInt(ctrl)
        if value is None:
            return None

        normalized = int(value)
        if minimum is not None:
            normalized = max(int(minimum), normalized)
        if maximum is not None:
            normalized = min(int(maximum), normalized)

        self._in_param_commit = True
        try:
            self.params[key] = normalized
            self._setCtrlValueSilently(ctrl, normalized)
        finally:
            self._in_param_commit = False
        return normalized

    def OnLearningRate(self, event):
        self._commit_float_param(self.learning_rate_ctrl, "learning_rate")

    def OnError(self, event):
        self._commit_float_param(self.error_ctrl, "error")

    def OnActivationIncrease(self, event):
        self._commit_float_param(self.activation_increase_ctrl, "activation_increase")

    def OnActivationLimit(self, event):
        self._commit_float_param(self.activation_limit_ctrl, "activation_limit")

    def OnSigmoidScale(self, event):
        self._commit_float_param(self.sigmoid_scale_ctrl, "sigmoid_scale")

    def OnWordChangeThreshold(self, event):
        self._commit_float_param(self.word_change_threshold_ctrl, "word_change_threshold")

    def OnZoom(self, event):
        value = self._commit_float_param(self.zoom_ctrl, "zoom", minimum=0.001)
        if value is not None:
            self.makeWordCircle(self.words)

    def OnProcessInterval(self, event):
        if bool(self.params.get("process_interval_from_rhythm_bpm", False)):
            self._sync_process_interval_from_bpm(force=True)
            return
        value = self._commit_float_param(self.process_interval_ctrl, "process_interval", minimum=0.1, maximum=8.0)
        if value is None:
            return
        self._update_process_interval_display_label()
        try:
            transition_crossfade = min(0.08, max(0.01, float(self.params["process_interval"]) * 0.45))
            sanasyna.set_transition_crossfade(transition_crossfade)
        except Exception:
            pass

    def _process_interval_from_bpm(self, bpm):
        safe_bpm = min(300.0, max(0.01, float(bpm)))
        # Map rhythm BPM to a musically broad process interval range:
        # 0.01 BPM -> 8.0 s, 300 BPM -> 0.1 s.
        t = (safe_bpm - 0.01) / (300.0 - 0.01)
        t = min(1.0, max(0.0, t))
        return 8.0 * ((0.1 / 8.0) ** t)

    def _update_process_interval_display_label(self):
        if not hasattr(self, "process_interval_value_label") or self.process_interval_value_label is None:
            return
        interval = float(self.params.get("process_interval", 0.1))
        mode = "sync" if bool(self.params.get("process_interval_from_rhythm_bpm", False)) else "manual"
        self.process_interval_value_label.SetLabel(f"Current interval: {interval:.3f} s ({mode})")

    def _sync_process_interval_from_bpm(self, force=False):
        if not force and not bool(self.params.get("process_interval_from_rhythm_bpm", False)):
            return
        bpm = float(self.params.get("rhythm_mod_bpm", 108.0))
        interval = self._process_interval_from_bpm(bpm)
        self.params["process_interval"] = interval
        if hasattr(self, "process_interval_ctrl") and self.process_interval_ctrl is not None:
            self._setCtrlValueSilently(self.process_interval_ctrl, interval)
        self._update_process_interval_display_label()
        try:
            transition_crossfade = min(0.08, max(0.01, float(interval) * 0.45))
            sanasyna.set_transition_crossfade(transition_crossfade)
        except Exception:
            pass

    def _update_process_interval_mode_ui(self):
        is_synced = bool(self.params.get("process_interval_from_rhythm_bpm", False))
        if hasattr(self, "process_interval_ctrl") and self.process_interval_ctrl is not None:
            self.process_interval_ctrl.Enable(not is_synced)
        self._update_process_interval_display_label()

    def OnProcessIntervalBPMMode(self, event):
        if self._suppress_param_events:
            return
        self.params["process_interval_from_rhythm_bpm"] = bool(self.process_interval_bpm_sync_checkbox.GetValue())
        self._update_process_interval_mode_ui()
        if self.params["process_interval_from_rhythm_bpm"]:
            self._sync_process_interval_from_bpm(force=True)

    def OnMelodyFromOwnTime(self, event):
        enabled = self.melody_from_own_time_checkbox.GetValue()
        self.params["melody_from_own_time"] = enabled
        self.last_audio_sentence_signature = None
        self.audio_playing = False
        if enabled:
            # Stop any looping playback so the simulationStep gate (is_playing) can pass
            sanasyna.stop()

    def OnFullscreen(self, event):
        if self._suppress_param_events:
            return
        self._apply_fullscreen_mode(bool(self.fullscreen_checkbox.GetValue()))

    def OnSelectionExploration(self, event):
        self._commit_float_param(self.selection_exploration_ctrl, "selection_exploration", minimum=0.0, maximum=1.0)

    def OnSelectionTopK(self, event):
        self._commit_int_param(self.selection_top_k_ctrl, "selection_top_k", minimum=1)

    def OnJumpProbability(self, event):
        self._commit_float_param(self.jump_probability_ctrl, "jump_probability", minimum=0.0, maximum=1.0)

    def OnJumpRadius(self, event):
        self._commit_int_param(self.jump_radius_ctrl, "jump_radius", minimum=0)

    def OnUsePhonemeRhyme(self, event):
        if self._suppress_param_events:
            return
        self.params["use_phoneme_rhyme"] = bool(self.use_phoneme_rhyme_checkbox.GetValue())

    def OnRhymeWeight(self, event):
        self._commit_float_param(self.rhyme_weight_ctrl, "rhyme_weight", minimum=0.0, maximum=1.0)

    def OnRhymeMinSimilarity(self, event):
        self._commit_float_param(self.rhyme_min_similarity_ctrl, "rhyme_min_similarity", minimum=0.0, maximum=1.0)

    def OnRhymeStrategy(self, event):
        if self._suppress_param_events:
            return
        selected_label = self.rhyme_strategy_choice.GetStringSelection()
        self.params["rhyme_strategy"] = self._rhyme_strategy_key_from_label(selected_label)

    def OnRhymeTailBias(self, event):
        self._commit_float_param(self.rhyme_tail_bias_ctrl, "rhyme_tail_bias", minimum=0.0, maximum=1.0)

    def OnPiperTTS(self, event):
        if self._suppress_param_events:
            return
        self.params["piper_tts_on"] = bool(self.piper_tts_checkbox.GetValue())
        self.last_audio_sentence_signature = None

    def OnPiperModelPath(self, event):
        if self._suppress_param_events or self._in_param_commit:
            return
        self.params["piper_model_path"] = str(self.piper_model_ctrl.GetValue()).strip()

    def OnSynthVolume(self, event):
        val = int(self.synth_volume_slider.GetValue())
        self.params["synth_volume"] = val / 100.0
        self.synth_volume_label.SetLabel(f"Synth volume: {val}%")

    def _update_compressor_labels(self):
        threshold = int(round(float(self.params.get("compressor_threshold_db", -18.0))))
        ratio = float(self.params.get("compressor_ratio", 3.0))
        makeup = int(round(float(self.params.get("compressor_makeup_db", 6.0))))
        self.compressor_threshold_label.SetLabel(f"Compressor threshold: {threshold} dB")
        self.compressor_ratio_label.SetLabel(f"Compressor ratio: {ratio:.1f}:1")
        self.compressor_makeup_label.SetLabel(f"Compressor makeup gain: +{makeup} dB")

    def _apply_compressor_state(self):
        payload = {
            "enabled": bool(self.params.get("compressor_enabled", False)),
            "threshold_db": float(self.params.get("compressor_threshold_db", -18.0)),
            "ratio": float(self.params.get("compressor_ratio", 3.0)),
            "makeup_db": float(self.params.get("compressor_makeup_db", 6.0)),
            "attack_ms": 8.0,
            "release_ms": 140.0,
        }
        try:
            sanasyna.set_compressor(payload)
        except Exception:
            pass

    def OnCompressorEnabled(self, event):
        if self._suppress_param_events:
            return
        self.params["compressor_enabled"] = bool(self.compressor_enabled_checkbox.GetValue())
        self._apply_compressor_state()

    def OnCompressorThreshold(self, event):
        val = int(self.compressor_threshold_slider.GetValue())
        self.params["compressor_threshold_db"] = float(val)
        self._update_compressor_labels()
        self._apply_compressor_state()

    def OnCompressorRatio(self, event):
        val = int(self.compressor_ratio_slider.GetValue())
        self.params["compressor_ratio"] = float(val) / 10.0
        self._update_compressor_labels()
        self._apply_compressor_state()

    def OnCompressorMakeup(self, event):
        val = int(self.compressor_makeup_slider.GetValue())
        self.params["compressor_makeup_db"] = float(val)
        self._update_compressor_labels()
        self._apply_compressor_state()

    def OnPiperVolume(self, event):
        val = int(self.piper_volume_slider.GetValue())
        self.params["piper_volume"] = val / 100.0
        self.piper_volume_label.SetLabel(f"Piper volume: {val}%")

    def OnPiperModelBrowse(self, event):
        with wx.FileDialog(
            self.frame,
            "Select Piper model",
            wildcard="Piper ONNX model (*.onnx)|*.onnx|All files (*.*)|*.*",
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
        ) as file_dialog:
            if file_dialog.ShowModal() == wx.ID_CANCEL:
                return
            selected_path = file_dialog.GetPath()

        self.params["piper_model_path"] = str(selected_path).strip()
        self._setCtrlValueSilently(self.piper_model_ctrl, self.params["piper_model_path"])
        self._set_piper_status(f"Piper model: {os.path.basename(self.params['piper_model_path'])}")

    def _set_piper_status(self, message):
        if not hasattr(self, "piper_tts_status_label"):
            return
        try:
            wx.CallAfter(self.piper_tts_status_label.SetLabel, str(message))
        except Exception:
            pass

    def _speak_sentence_with_piper(self, sentence):
        text = str(sentence).strip()
        if text == "":
            return
        if not bool(self.params.get("piper_tts_on", False)):
            return

        piper_bin = shutil.which("piper")
        if piper_bin is not None:
            command = [piper_bin]
        else:
            command = [sys.executable, "-m", "piper"]

        model_path = str(self.params.get("piper_model_path", "")).strip()
        if model_path == "" or not os.path.isfile(model_path):
            self._set_piper_status("Piper TTS: set valid .onnx model path")
            return

        tmp_wav = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_wav = tmp_file.name

            result = subprocess.run(
                command + ["--model", model_path, "--output_file", tmp_wav],
                input=text.encode("utf-8"),
                capture_output=True,
                check=False,
                timeout=30,
            )
            if result.returncode != 0:
                stderr_text = result.stderr.decode("utf-8", errors="ignore").strip()
                if stderr_text == "":
                    stderr_text = "unknown synthesis error"
                self._set_piper_status(f"Piper TTS error: {stderr_text}")
                return

            with wave.open(tmp_wav, "rb") as wav_file:
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                sample_rate = wav_file.getframerate()
                raw_frames = wav_file.readframes(wav_file.getnframes())

            if sample_width != 2:
                self._set_piper_status("Piper TTS failed: unsupported WAV sample width")
                return

            audio_data = np.frombuffer(raw_frames, dtype=np.int16)
            if channels > 1:
                audio_data = audio_data.reshape(-1, channels)
            audio_data = audio_data.astype(np.float32) / 32768.0

            piper_vol = min(1.0, max(0.0, float(self.params.get("piper_volume", 0.5))))
            audio_data = audio_data * piper_vol

            # If a newer finalized sentence arrived while this one was being
            # synthesized, skip stale playback and let the worker continue.
            with self._piper_queue_lock:
                has_newer_pending = bool(self._piper_sentence_queue) and self._last_queued_sentence != text
            if has_newer_pending:
                return

            self._set_piper_status("Piper TTS: speaking")
            if not sanasyna.play_overlay_samples(audio_data, sample_rate=sample_rate, wait=True):
                self._set_piper_status("Piper TTS failed: shared audio backend unavailable")
                return
            self._last_spoken_sentence = text
        except Exception as error:
            self._set_piper_status(f"Piper TTS failed: {error}")
        finally:
            if tmp_wav is not None:
                try:
                    os.remove(tmp_wav)
                except Exception:
                    pass

    def _ensure_piper_worker(self):
        if self._piper_speak_thread is not None and self._piper_speak_thread.is_alive():
            return

        self._piper_worker_stop = False
        self._piper_speak_thread = threading.Thread(
            target=self._piper_queue_worker,
            daemon=True,
        )
        self._piper_speak_thread.start()

    def _piper_queue_worker(self):
        while True:
            self._piper_queue_event.wait()

            while True:
                sentence = None
                should_stop = False
                with self._piper_queue_lock:
                    if self._piper_sentence_queue:
                        sentence = self._piper_sentence_queue.popleft()
                    else:
                        should_stop = self._piper_worker_stop
                        self._piper_queue_event.clear()

                if sentence is None:
                    if should_stop:
                        return
                    break

                if not self.running or self.closed:
                    continue

                self._speak_sentence_with_piper(sentence)

    def _speak_sentence_with_piper_async(self, sentence):
        text = str(sentence).strip()
        if text == "":
            return
        if not bool(self.params.get("piper_tts_on", False)):
            return

        with self._piper_queue_lock:
            if text == self._last_spoken_sentence or text == self._last_queued_sentence:
                return
            # Keep only the latest finalized sentence so speech stays aligned
            # with the sentence shown in the network view and output file.
            self._piper_sentence_queue.clear()
            self._piper_sentence_queue.append(text)
            self._last_queued_sentence = text

        self._ensure_piper_worker()
        self._piper_queue_event.set()

    def OnImportMode(self, event):
        if self._suppress_param_events:
            return
        selected_mode = self.import_mode_choice.GetStringSelection()
        if selected_mode == "Replace database":
            self.params["import_mode"] = "replace"
        else:
            self.params["import_mode"] = "append"

    def OnAudioWaveMode(self, event):
        if self._suppress_param_events:
            return
        selected_mode = self.audio_wave_mode_choice.GetStringSelection()
        _wave_label_map = {
            "Pure sine": "pure_sine",
            "Noise-heavy": "noise_heavy",
            "Classic analog": "classic_analog",
            "Neuro formant": "neuro_formant",
            "Neuro pulse": "neuro_pulse",
            "Neuro ring": "neuro_ring",
            "Neuro fold": "neuro_fold",
            "Neuro FM": "neuro_fm",
        }
        self.params["audio_wave_mode"] = _wave_label_map.get(selected_mode, "dynamic")
        self.last_audio_sentence_signature = None

    def OnFrequencyMappingMode(self, event):
        if self._suppress_param_events:
            return
        selected_label = self.frequency_mapping_choice.GetStringSelection()
        self.params["frequency_mapping_mode"] = self._frequency_mapping_key_from_label(selected_label)
        self.last_audio_sentence_signature = None

    def OnVoiceCount(self, event):
        if self._suppress_param_events:
            return
        selected_voice_count = self.voice_count_choice.GetStringSelection()
        try:
            self.params["voice_count"] = max(1, min(4, int(selected_voice_count)))
        except Exception:
            self.params["voice_count"] = 1
        self.params["rhythm_style"] = "manual"
        if hasattr(self, "rhythm_style_choice"):
            self.rhythm_style_choice.SetStringSelection(self._rhythm_style_label_from_key("manual"))
        self.last_audio_sentence_signature = None

    def OnRhythmStyle(self, event):
        if self._suppress_param_events:
            return
        selected_label = self.rhythm_style_choice.GetStringSelection()
        style_key = self._rhythm_style_key_from_label(selected_label)
        self.params["rhythm_style"] = style_key
        self._apply_rhythm_style_preset(style_key)
        self._sync_controls_from_params()
        self.last_audio_sentence_signature = None

    def OnBeatLibraryStyle(self, event):
        if self._suppress_param_events:
            return
        selected_label = self.beat_library_style_choice.GetStringSelection()
        self.params["beat_library_style"] = self._beat_library_key_from_label(selected_label)
        self.params["rhythm_style"] = "manual"
        self.rhythm_style_choice.SetStringSelection(self._rhythm_style_label_from_key("manual"))
        self.last_audio_sentence_signature = None

    def OnStrictCounterpoint(self, event):
        if self._suppress_param_events:
            return
        self.params["strict_counterpoint"] = bool(self.strict_counterpoint_checkbox.GetValue())
        self.last_audio_sentence_signature = None

    def OnVoiceSpread(self, event):
        self._commit_float_param(self.voice_spread_ctrl, "voice_spread", minimum=0.3, maximum=5.0)
        self.last_audio_sentence_signature = None

    def OnVoiceDistance(self, event):
        self._commit_float_param(self.voice_distance_ctrl, "voice_distance", minimum=0.0, maximum=1.0)
        self.last_audio_sentence_signature = None

    def OnVoiceDistanceContext(self, event):
        self._commit_int_param(self.voice_distance_context_ctrl, "voice_distance_context", minimum=1, maximum=32)
        self.params["rhythm_style"] = "manual"
        if hasattr(self, "rhythm_style_choice"):
            self.rhythm_style_choice.SetStringSelection(self._rhythm_style_label_from_key("manual"))
        self.last_audio_sentence_signature = None

    def OnRhythmicDivergence(self, event):
        self._commit_float_param(self.rhythmic_divergence_ctrl, "rhythmic_divergence", minimum=0.0, maximum=1.0)
        self.params["rhythm_style"] = "manual"
        if hasattr(self, "rhythm_style_choice"):
            self.rhythm_style_choice.SetStringSelection(self._rhythm_style_label_from_key("manual"))
        self.last_audio_sentence_signature = None

    def OnRhythmGateStrength(self, event):
        self._commit_float_param(self.rhythm_gate_strength_ctrl, "rhythm_gate_strength", minimum=0.0, maximum=1.0)
        self.params["rhythm_style"] = "manual"
        if hasattr(self, "rhythm_style_choice"):
            self.rhythm_style_choice.SetStringSelection(self._rhythm_style_label_from_key("manual"))
        self.last_audio_sentence_signature = None

    def OnRhythmStretchStrength(self, event):
        self._commit_float_param(self.rhythm_stretch_strength_ctrl, "rhythm_stretch_strength", minimum=0.0, maximum=1.0)
        self.params["rhythm_style"] = "manual"
        if hasattr(self, "rhythm_style_choice"):
            self.rhythm_style_choice.SetStringSelection(self._rhythm_style_label_from_key("manual"))
        self.last_audio_sentence_signature = None

    def OnRhythmRotation(self, event):
        self._commit_int_param(self.rhythm_rotation_ctrl, "rhythm_rotation", minimum=0, maximum=31)
        self.params["rhythm_style"] = "manual"
        if hasattr(self, "rhythm_style_choice"):
            self.rhythm_style_choice.SetStringSelection(self._rhythm_style_label_from_key("manual"))
        self.last_audio_sentence_signature = None

    def OnRhythmRadicality(self, event):
        self._commit_float_param(self.rhythm_radicality_ctrl, "rhythm_radicality", minimum=0.0, maximum=1.0)
        self.params["rhythm_style"] = "manual"
        if hasattr(self, "rhythm_style_choice"):
            self.rhythm_style_choice.SetStringSelection(self._rhythm_style_label_from_key("manual"))
        self.last_audio_sentence_signature = None

    def OnRhythmModBPM(self, event):
        self._commit_float_param(self.rhythm_mod_bpm_ctrl, "rhythm_mod_bpm", minimum=0.01, maximum=300.0)
        self._sync_process_interval_from_bpm()
        self._apply_rhythm_modulation_state()

    def OnAdditiveRhythmWeight(self, event):
        val = int(self.additive_rhythm_weight_slider.GetValue())
        self.params["additive_rhythm_weight"] = val / 100.0
        self.additive_rhythm_weight_label.SetLabel(f"Additive rhythm weight: {val}%")
        self._apply_rhythm_modulation_state()

    def OnDivisiveRhythmWeight(self, event):
        val = int(self.divisive_rhythm_weight_slider.GetValue())
        self.params["divisive_rhythm_weight"] = val / 100.0
        self.divisive_rhythm_weight_label.SetLabel(f"Divisive rhythm weight: {val}%")
        self._apply_rhythm_modulation_state()

    def OnDivisiveRhythmSignature(self, event):
        if self._suppress_param_events:
            return
        selected_label = self.divisive_signature_choice.GetStringSelection()
        self.params["divisive_rhythm_signature"] = self._divisive_signature_key_from_label(selected_label)
        self._apply_rhythm_modulation_state()

    def _update_additive_blocks(self, blocks):
        cleaned = []
        for value in blocks:
            try:
                block = int(float(value))
            except Exception:
                continue
            if block > 0:
                cleaned.append(block)
        self.params["additive_rhythm_blocks"] = cleaned
        self._update_additive_pattern_status_label()
        self._apply_rhythm_modulation_state()
        self._refresh_additive_editor_sequence()

    def OnOpenAdditiveRhythmEditor(self, event):
        if self.additive_editor_frame is not None and self.additive_editor_frame.IsShown():
            self.additive_editor_frame.Raise()
            return

        frame = wx.Frame(self.frame, -1, "Additive Rhythm Editor", size=(960, 320))
        frame.Bind(wx.EVT_CLOSE, self._on_additive_editor_close)

        panel = wx.Panel(frame, -1)
        sizer = wx.BoxSizer(wx.VERTICAL)

        helper = wx.StaticText(panel, -1, "Append blocks from the palette, then move blocks left/right or delete them.")
        sizer.Add(helper, 0, wx.LEFT | wx.RIGHT | wx.TOP, 8)

        palette_sizer = wx.WrapSizer(wx.HORIZONTAL)
        for block in self._additive_block_library():
            button = wx.Button(panel, -1, f"+ {block}")
            button.Bind(wx.EVT_BUTTON, lambda evt, b=block: self._append_additive_block(b))
            palette_sizer.Add(button, 0, wx.ALL, 3)
        clear_button = wx.Button(panel, -1, "Clear all")
        clear_button.Bind(wx.EVT_BUTTON, lambda evt: self._update_additive_blocks([]))
        palette_sizer.Add(clear_button, 0, wx.ALL, 3)
        sizer.Add(palette_sizer, 0, wx.LEFT | wx.RIGHT | wx.TOP, 6)

        self.additive_sequence_scroll = wx.ScrolledWindow(panel, -1, style=wx.HSCROLL | wx.VSCROLL | wx.BORDER_SIMPLE)
        self.additive_sequence_scroll.SetScrollRate(20, 20)
        self.additive_sequence_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.additive_sequence_scroll.SetSizer(self.additive_sequence_sizer)
        sizer.Add(self.additive_sequence_scroll, 1, wx.EXPAND | wx.ALL, 8)

        self.additive_timeline_preview = RhythmTimelinePreviewPanel(panel)
        sizer.Add(self.additive_timeline_preview, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)

        panel.SetSizer(sizer)
        self.additive_editor_frame = frame
        self._refresh_additive_editor_sequence()
        self._refresh_additive_timeline_preview()
        frame.Show()

    def _on_additive_editor_close(self, event):
        self.additive_editor_frame = None
        self.additive_sequence_scroll = None
        self.additive_sequence_sizer = None
        self.additive_timeline_preview = None
        if event is not None:
            event.Skip()

    def _append_additive_block(self, block):
        blocks = list(self._normalized_additive_blocks())
        blocks.append(int(block))
        self._update_additive_blocks(blocks)

    def _move_additive_block(self, index, direction):
        blocks = list(self._normalized_additive_blocks())
        target = index + direction
        if index < 0 or index >= len(blocks):
            return
        if target < 0 or target >= len(blocks):
            return
        blocks[index], blocks[target] = blocks[target], blocks[index]
        self._update_additive_blocks(blocks)

    def _delete_additive_block(self, index):
        blocks = list(self._normalized_additive_blocks())
        if index < 0 or index >= len(blocks):
            return
        del blocks[index]
        self._update_additive_blocks(blocks)

    def _refresh_additive_editor_sequence(self):
        if self.additive_sequence_sizer is None or self.additive_sequence_scroll is None:
            return

        self.additive_sequence_sizer.Clear(True)
        blocks = self._normalized_additive_blocks()

        if not blocks:
            empty_label = wx.StaticText(self.additive_sequence_scroll, -1, "Additive chain is empty. Use + buttons above to append blocks.")
            self.additive_sequence_sizer.Add(empty_label, 0, wx.ALL, 8)
            self.additive_sequence_scroll.Layout()
            self.additive_sequence_scroll.FitInside()
            return

        for index, block in enumerate(blocks):
            card = wx.Panel(self.additive_sequence_scroll, -1)
            card_sizer = wx.BoxSizer(wx.VERTICAL)
            card_label = wx.StaticText(card, -1, f"{index + 1}: {block}")
            button_row = wx.BoxSizer(wx.HORIZONTAL)
            left_btn = wx.Button(card, -1, "<-", size=(34, 26))
            right_btn = wx.Button(card, -1, "->", size=(34, 26))
            del_btn = wx.Button(card, -1, "x", size=(30, 26))

            left_btn.Bind(wx.EVT_BUTTON, lambda evt, i=index: self._move_additive_block(i, -1))
            right_btn.Bind(wx.EVT_BUTTON, lambda evt, i=index: self._move_additive_block(i, 1))
            del_btn.Bind(wx.EVT_BUTTON, lambda evt, i=index: self._delete_additive_block(i))

            button_row.Add(left_btn, 0, wx.RIGHT, 2)
            button_row.Add(right_btn, 0, wx.RIGHT, 2)
            button_row.Add(del_btn, 0)

            card_sizer.Add(card_label, 0, wx.ALIGN_CENTER | wx.BOTTOM, 4)
            card_sizer.Add(button_row, 0, wx.ALIGN_CENTER)
            card.SetSizer(card_sizer)
            self.additive_sequence_sizer.Add(card, 0, wx.ALL, 5)

        self.additive_sequence_scroll.Layout()
        self.additive_sequence_scroll.FitInside()

    def OnMelodyCoherence(self, event):
        self._commit_float_param(self.melody_coherence_ctrl, "melody_coherence", minimum=0.0, maximum=1.0)
        self.last_audio_sentence_signature = None

    def OnMelodySpeed(self, event):
        self._commit_float_param(self.melody_speed_ctrl, "melody_speed", minimum=0.2, maximum=6.0)
        self.params["rhythm_style"] = "manual"
        if hasattr(self, "rhythm_style_choice"):
            self.rhythm_style_choice.SetStringSelection(self._rhythm_style_label_from_key("manual"))
        self.last_audio_sentence_signature = None

    def OnMinNoteDuration(self, event):
        self._commit_float_param(self.min_note_duration_ctrl, "min_note_duration", minimum=0.01, maximum=1.0)
        self.params["rhythm_style"] = "manual"
        if hasattr(self, "rhythm_style_choice"):
            self.rhythm_style_choice.SetStringSelection(self._rhythm_style_label_from_key("manual"))
        self.last_audio_sentence_signature = None

    def _applyADSRToAudio(self):
        sanasyna.set_adsr(
            self.params["adsr_attack"],
            self.params["adsr_decay"],
            self.params["adsr_sustain"],
            self.params["adsr_release"],
        )
        self.adsr_display.set_adsr(
            self.params["adsr_attack"],
            self.params["adsr_decay"],
            self.params["adsr_sustain"],
            self.params["adsr_release"],
        )

    def OnADSRAttack(self, event):
        value = self._commit_float_param(self.adsr_attack_ctrl, "adsr_attack", minimum=0.0)
        if value is not None:
            self._applyADSRToAudio()

    def OnADSRDecay(self, event):
        value = self._commit_float_param(self.adsr_decay_ctrl, "adsr_decay", minimum=0.0)
        if value is not None:
            self._applyADSRToAudio()

    def OnADSRSustain(self, event):
        value = self._commit_float_param(self.adsr_sustain_ctrl, "adsr_sustain", minimum=0.0, maximum=1.0)
        if value is not None:
            self._applyADSRToAudio()

    def OnADSRRelease(self, event):
        value = self._commit_float_param(self.adsr_release_ctrl, "adsr_release", minimum=0.0)
        if value is not None:
            self._applyADSRToAudio()

    def OnADSR(self, event):
        event_ctrl = None
        try:
            event_ctrl = event.GetEventObject()
        except Exception:
            event_ctrl = None

        if event_ctrl == self.adsr_attack_ctrl:
            self.OnADSRAttack(event)
        elif event_ctrl == self.adsr_decay_ctrl:
            self.OnADSRDecay(event)
        elif event_ctrl == self.adsr_sustain_ctrl:
            self.OnADSRSustain(event)
        elif event_ctrl == self.adsr_release_ctrl:
            self.OnADSRRelease(event)

    def OnClose(self, event):
        if self.closed:
            return

        self.running = False
        self.closed = True
        if self.output_timer is not None:
            self.output_timer.Stop()
            self.output_timer = None
        if self.timer is not None:
            self.timer.Stop()
            self.timer = None
        self._piper_worker_stop = True
        with self._piper_queue_lock:
            self._piper_sentence_queue.clear()
        self._piper_queue_event.set()
        try:
            if self._piper_audio_channel is not None and self._piper_audio_channel.get_busy():
                self._piper_audio_channel.stop()
        except Exception:
            pass
        sanasyna.stop()
        sanasyna.close()
        pygame.quit()
        if hasattr(self, "outfile") and not self.outfile.closed:
            self.outfile.close()
        if self.output_frame is not None:
            self.output_frame.Destroy()
            self.output_frame = None
        if self.additive_editor_frame is not None:
            self.additive_editor_frame.Destroy()
            self.additive_editor_frame = None
            self.additive_sequence_scroll = None
            self.additive_sequence_sizer = None
            self.additive_timeline_preview = None
        if self.frame is not None:
            self.frame.Destroy()
        if self.app is not None and self.app.IsMainLoopRunning():
            self.app.ExitMainLoop()

    def getParam(self, param):
        return self.params[param]

    def setParam(self, param, value):
        self.params[param] = value

    def _get_display_window(self):
        if pygame_sdl2_video is None:
            return None
        try:
            self._pygame_window = pygame_sdl2_video.Window.from_display_module()
        except Exception:
            self._pygame_window = None
        return self._pygame_window

    def _is_effective_fullscreen(self):
        if bool(self.params.get("fullscreen", False)):
            return True
        if not pygame.display.get_init():
            return False
        try:
            current_size = tuple(pygame.display.get_window_size())
        except Exception:
            return False
        info = pygame.display.Info()
        display_size = (
            max(1, int(getattr(info, "current_w", 0) or 0)),
            max(1, int(getattr(info, "current_h", 0) or 0)),
        )
        return current_size == display_size and current_size != tuple(self._windowed_size)

    def _shift_network_to_window_center(self, previous_size, current_size):
        if not self.words:
            return
        if previous_size is None or current_size is None:
            return

        old_width, old_height = float(previous_size[0]), float(previous_size[1])
        new_width, new_height = float(current_size[0]), float(current_size[1])
        delta_x = (new_width - old_width) / 2.0
        delta_y = (new_height - old_height) / 2.0

        if delta_x == 0.0 and delta_y == 0.0:
            return

        for word in self.words:
            word.x += delta_x
            word.y += delta_y
            word.neuron.x += delta_x
            word.neuron.y += delta_y

    def _sync_display_window_state(self):
        if not pygame.display.get_init():
            return
        previous_size = tuple(self.size) if self.size is not None else None
        try:
            current_size = tuple(pygame.display.get_window_size())
        except Exception:
            current_size = tuple(self.size) if self.size is not None else tuple(self._windowed_size)

        effective_fullscreen = self._is_effective_fullscreen()
        self.params["fullscreen"] = effective_fullscreen
        self.size = current_size

        if previous_size is not None and previous_size != current_size and effective_fullscreen:
            self._shift_network_to_window_center(previous_size, current_size)

        if not effective_fullscreen:
            self._windowed_size = current_size

        if hasattr(self, "fullscreen_checkbox") and self.fullscreen_checkbox is not None:
            self.fullscreen_checkbox.SetValue(effective_fullscreen)

    def _apply_fullscreen_mode(self, enabled):
        enabled = bool(enabled)
        flags = pygame.RESIZABLE

        if self.screen is None:
            self.screen = pygame.display.set_mode(tuple(self._windowed_size), flags)

        window = self._get_display_window()

        if enabled:
            self._sync_display_window_state()
            if not self.params.get("fullscreen", False) and self.size is not None:
                self._windowed_size = tuple(self.size)

            if window is not None:
                try:
                    window.resizable = True
                except Exception:
                    pass
                window.set_fullscreen(desktop=True)
                try:
                    self.size = tuple(window.size)
                except Exception:
                    self.size = tuple(pygame.display.get_window_size())
                self.screen = pygame.display.get_surface()
            else:
                info = pygame.display.Info()
                self.size = (
                    max(1, int(getattr(info, "current_w", 0) or self._windowed_size[0])),
                    max(1, int(getattr(info, "current_h", 0) or self._windowed_size[1])),
                )
                self.screen = pygame.display.set_mode(self.size, pygame.FULLSCREEN)
        else:
            if window is not None:
                window.set_windowed()
                try:
                    window.size = tuple(self._windowed_size)
                except Exception:
                    pass
                self.screen = pygame.display.get_surface()
            else:
                self.screen = pygame.display.set_mode(tuple(self._windowed_size), flags)
            self.size = tuple(self._windowed_size)

        self.params["fullscreen"] = enabled

        pygame.display.set_caption("SanaVerkko")

        if hasattr(self, "fullscreen_checkbox") and self.fullscreen_checkbox is not None:
            self.fullscreen_checkbox.SetValue(enabled)

    def initPygame(self):
        pygame.init()
        self._apply_fullscreen_mode(bool(self.params.get("fullscreen", False)))
        self.clock = pygame.time.Clock()

    def initAudio(self):
        self.audio_sample_rate = 22050
        self.audio_refresh_interval = 0.15
        self.last_audio_update = 0
        self.audio_playing = False
        self.audio_wave_index = 0
        self.audio_waveforms = ["sine", "triangle", "square", "sawtooth", "noise"]
        self._synthesis_thread = None
        self.last_audio_sentence_signature = None
        self.frequency_mapping_cache = {}
        self.frequency_mapping_boundaries = {}
        self.last_frequency_mapping_info = []
        sanasyna.init_audio(self.audio_sample_rate)
        try:
            sanasyna.set_transition_crossfade(0.03)
        except Exception:
            pass
        self._apply_rhythm_modulation_state()
        self._apply_compressor_state()

    def _frequency_mapping_modes(self):
        return [
            ("original_notes", "Original notes"),
            ("pythagorean_pentatonic", "Pythagorean pentatonic"),
            ("pythagorean_8_note", "Pythagorean 8 note"),
            ("just_intonation_5_limit", "Just intonation 5-limit"),
            ("equal_tempered_ionian", "Equal tempered ionian"),
            ("equal_tempered_dorian", "Equal tempered dorian"),
            ("equal_tempered_frygian", "Equal tempered frygian"),
            ("equal_tempered_lydian", "Equal tempered lydian"),
            ("equal_tempered_mixolydian", "Equal tempered mixolydian"),
            ("equal_tempered_aeolian", "Equal tempered aeolian"),
            ("equal_tempered_locrian", "Equal tempered locrian"),
            ("equal_tempered_12_note", "Equal tempered 12 note"),
            ("equal_tempered_19_note", "Equal tempered 19 note"),
            ("equal_tempered_24_note", "Equal tempered 24 note"),
            ("equal_tempered_31_note", "Equal tempered 31 note"),
            ("equal_tempered_36_note", "Equal tempered 36 note"),
            ("equal_tempered_48_note", "Equal tempered 48 note"),
        ]

    def _rhythm_style_modes(self):
        return [
            ("manual", "Manual"),
            ("tight_pulse", "Tight pulse"),
            ("hocket_duo", "Hocket duo"),
            ("polyrhythm_trio", "Polyrhythm trio"),
            ("fractured_quartet", "Fractured quartet"),
            ("hyperbanana", "Hyperbanana cadence"),
        ]

    def _beat_library_modes(self):
        return [
            ("auto", "Auto"),
            ("duo_cross", "Duo cross"),
            ("duo_hocket", "Duo hocket"),
            ("trio_3over2", "Trio 3 over 2"),
            ("trio_tresillo", "Trio tresillo"),
            ("quartet_grid_fracture", "Quartet grid fracture"),
            ("quartet_hyperbanana", "Quartet hyperbanana"),
        ]

    def _divisive_signature_modes(self):
        return [
            ("2/4", "2/4"),
            ("3/4", "3/4"),
            ("4/4", "4/4"),
            ("5/4", "5/4"),
            ("6/8", "6/8"),
            ("7/8", "7/8"),
            ("9/8", "9/8"),
            ("12/8", "12/8"),
        ]

    def _divisive_signature_label_from_key(self, signature_key):
        for key, label in self._divisive_signature_modes():
            if key == signature_key:
                return label
        return "4/4"

    def _divisive_signature_key_from_label(self, signature_label):
        for key, label in self._divisive_signature_modes():
            if label == signature_label:
                return key
        return "4/4"

    def _additive_block_library(self):
        return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16]

    def _normalized_additive_blocks(self):
        raw_blocks = self.params.get("additive_rhythm_blocks", [])
        normalized = []
        if isinstance(raw_blocks, (list, tuple)):
            for value in raw_blocks:
                try:
                    block = int(float(value))
                except Exception:
                    continue
                if block > 0:
                    normalized.append(block)
        self.params["additive_rhythm_blocks"] = normalized
        return normalized

    def _update_additive_pattern_status_label(self):
        blocks = self._normalized_additive_blocks()
        pattern = " + ".join(str(block) for block in blocks) if blocks else "(empty chain)"
        total_pulses = sum(blocks)
        if hasattr(self, "additive_pattern_status") and self.additive_pattern_status is not None:
            self.additive_pattern_status.SetLabel(f"Additive pattern: {pattern} (sum {total_pulses})")

    def _refresh_additive_timeline_preview(self):
        if self.additive_timeline_preview is None:
            return
        self.additive_timeline_preview.set_rhythm_state(
            self._normalized_additive_blocks(),
            self.params.get("divisive_rhythm_signature", "4/4"),
            self.params.get("additive_rhythm_weight", 0.0),
            self.params.get("divisive_rhythm_weight", 0.0),
        )

    def _apply_rhythm_modulation_state(self):
        blocks = self._normalized_additive_blocks()
        signature = str(self.params.get("divisive_rhythm_signature", "4/4"))
        if signature not in {key for key, _ in self._divisive_signature_modes()}:
            signature = "4/4"
            self.params["divisive_rhythm_signature"] = signature

        payload = {
            "bpm": float(self.params.get("rhythm_mod_bpm", 108.0)),
            "additive_blocks": blocks,
            "additive_weight": float(self.params.get("additive_rhythm_weight", 0.0)),
            "divisive_signature": signature,
            "divisive_weight": float(self.params.get("divisive_rhythm_weight", 0.0)),
        }
        try:
            sanasyna.set_rhythm_modulators(payload)
        except Exception:
            pass
        self._refresh_additive_timeline_preview()

    def _beat_library_label_from_key(self, style_key):
        for key, label in self._beat_library_modes():
            if key == style_key:
                return label
        return "Auto"

    def _beat_library_key_from_label(self, style_label):
        for key, label in self._beat_library_modes():
            if label == style_label:
                return key
        return "auto"

    def _rhyme_strategy_modes(self):
        return [
            ("suffix", "Suffix only"),
            ("whole_word", "Whole word"),
            ("hybrid", "Hybrid"),
        ]

    def _rhyme_strategy_label_from_key(self, strategy_key):
        for key, label in self._rhyme_strategy_modes():
            if key == strategy_key:
                return label
        return "Hybrid"

    def _rhyme_strategy_key_from_label(self, strategy_label):
        for key, label in self._rhyme_strategy_modes():
            if label == strategy_label:
                return key
        return "hybrid"

    def _rhythm_style_label_from_key(self, style_key):
        for key, label in self._rhythm_style_modes():
            if key == style_key:
                return label
        return "Manual"

    def _rhythm_style_key_from_label(self, style_label):
        for key, label in self._rhythm_style_modes():
            if label == style_label:
                return key
        return "manual"

    def _apply_rhythm_style_preset(self, style_key):
        presets = {
            "manual": {},
            "tight_pulse": {
                "voice_count": 2,
                "rhythmic_divergence": 0.20,
                "voice_distance_context": 3,
                "beat_library_style": "duo_cross",
                "rhythm_gate_strength": 0.20,
                "rhythm_stretch_strength": 0.35,
                "rhythm_rotation": 0,
                "rhythm_radicality": 0.20,
                "melody_speed": 1.10,
                "min_note_duration": 0.04,
            },
            "hocket_duo": {
                "voice_count": 2,
                "rhythmic_divergence": 0.75,
                "voice_distance_context": 5,
                "beat_library_style": "duo_hocket",
                "rhythm_gate_strength": 0.85,
                "rhythm_stretch_strength": 0.75,
                "rhythm_rotation": 1,
                "rhythm_radicality": 0.70,
                "melody_speed": 1.05,
                "min_note_duration": 0.03,
            },
            "polyrhythm_trio": {
                "voice_count": 3,
                "rhythmic_divergence": 0.85,
                "voice_distance_context": 6,
                "beat_library_style": "trio_3over2",
                "rhythm_gate_strength": 0.90,
                "rhythm_stretch_strength": 0.85,
                "rhythm_rotation": 2,
                "rhythm_radicality": 0.82,
                "melody_speed": 1.0,
                "min_note_duration": 0.025,
            },
            "fractured_quartet": {
                "voice_count": 4,
                "rhythmic_divergence": 0.92,
                "voice_distance_context": 7,
                "beat_library_style": "quartet_grid_fracture",
                "rhythm_gate_strength": 1.0,
                "rhythm_stretch_strength": 0.90,
                "rhythm_rotation": 3,
                "rhythm_radicality": 0.92,
                "melody_speed": 0.95,
                "min_note_duration": 0.02,
            },
            "hyperbanana": {
                "voice_count": 4,
                "rhythmic_divergence": 1.0,
                "voice_distance_context": 8,
                "beat_library_style": "quartet_hyperbanana",
                "rhythm_gate_strength": 1.0,
                "rhythm_stretch_strength": 1.0,
                "rhythm_rotation": 4,
                "rhythm_radicality": 1.0,
                "melody_speed": 1.2,
                "min_note_duration": 0.015,
            },
        }

        preset = presets.get(style_key, {})
        for key, value in preset.items():
            self.params[key] = value

    def _frequency_mapping_label_from_key(self, mode_key):
        for key, label in self._frequency_mapping_modes():
            if key == mode_key:
                return label
        return "Original notes"

    def _frequency_mapping_key_from_label(self, mode_label):
        for key, label in self._frequency_mapping_modes():
            if label == mode_label:
                return key
        return "original_notes"

    def _frequency_mapping_definition(self, mode_key):
        definitions = {
            "pythagorean_pentatonic": {
                "kind": "ratio",
                "ratios": [1.0, 9.0 / 8.0, 81.0 / 64.0, 3.0 / 2.0, 27.0 / 16.0],
                "prefix": "PYPEN",
            },
            "pythagorean_8_note": {
                "kind": "ratio",
                "ratios": [1.0, 9.0 / 8.0, 81.0 / 64.0, 4.0 / 3.0, 3.0 / 2.0, 27.0 / 16.0, 243.0 / 128.0, 2.0],
                "prefix": "PY8",
            },
            "just_intonation_5_limit": {
                "kind": "ratio",
                "ratios": [1.0, 9.0 / 8.0, 5.0 / 4.0, 4.0 / 3.0, 3.0 / 2.0, 5.0 / 3.0, 15.0 / 8.0, 2.0],
                "prefix": "JI5",
            },
            "equal_tempered_ionian": {
                "kind": "et_mode",
                "steps": [0, 2, 4, 5, 7, 9, 11],
                "divisions": 12,
                "prefix": "ET_ION",
            },
            "equal_tempered_dorian": {
                "kind": "et_mode",
                "steps": [0, 2, 3, 5, 7, 9, 10],
                "divisions": 12,
                "prefix": "ET_DOR",
            },
            "equal_tempered_frygian": {
                "kind": "et_mode",
                "steps": [0, 1, 3, 5, 7, 8, 10],
                "divisions": 12,
                "prefix": "ET_FRY",
            },
            "equal_tempered_lydian": {
                "kind": "et_mode",
                "steps": [0, 2, 4, 6, 7, 9, 11],
                "divisions": 12,
                "prefix": "ET_LYD",
            },
            "equal_tempered_mixolydian": {
                "kind": "et_mode",
                "steps": [0, 2, 4, 5, 7, 9, 10],
                "divisions": 12,
                "prefix": "ET_MIX",
            },
            "equal_tempered_aeolian": {
                "kind": "et_mode",
                "steps": [0, 2, 3, 5, 7, 8, 10],
                "divisions": 12,
                "prefix": "ET_AEO",
            },
            "equal_tempered_locrian": {
                "kind": "et_mode",
                "steps": [0, 1, 3, 5, 6, 8, 10],
                "divisions": 12,
                "prefix": "ET_LOC",
            },
            "equal_tempered_12_note": {
                "kind": "et_full",
                "divisions": 12,
                "prefix": "ET12",
            },
            "equal_tempered_19_note": {
                "kind": "et_full",
                "divisions": 19,
                "prefix": "ET19",
            },
            "equal_tempered_24_note": {
                "kind": "et_full",
                "divisions": 24,
                "prefix": "ET24",
            },
            "equal_tempered_31_note": {
                "kind": "et_full",
                "divisions": 31,
                "prefix": "ET31",
            },
            "equal_tempered_36_note": {
                "kind": "et_full",
                "divisions": 36,
                "prefix": "ET36",
            },
            "equal_tempered_48_note": {
                "kind": "et_full",
                "divisions": 48,
                "prefix": "ET48",
            },
        }
        return definitions.get(mode_key)

    def _build_frequency_mapping_entries(self, mode_key, min_freq=40.0, max_freq=4000.0, root_freq=110.0):
        definition = self._frequency_mapping_definition(mode_key)
        if definition is None:
            return []

        tones = []
        kind = definition["kind"]

        if kind == "ratio":
            ratios = definition["ratios"]
            for octave in range(-8, 9):
                octave_ratio = 2.0 ** octave
                for degree, ratio in enumerate(ratios):
                    freq = root_freq * ratio * octave_ratio
                    if min_freq * 0.5 <= freq <= max_freq * 2.0:
                        tones.append((freq, degree, octave))
        else:
            divisions = int(definition["divisions"])
            if kind == "et_mode":
                steps = definition["steps"]
            else:
                steps = list(range(divisions))

            for octave in range(-8, 9):
                for degree, step in enumerate(steps):
                    freq = root_freq * (2.0 ** (octave + (float(step) / float(divisions))))
                    if min_freq * 0.5 <= freq <= max_freq * 2.0:
                        tones.append((freq, degree, octave))

        tones = sorted(tones, key=lambda item: item[0])
        if len(tones) < 2:
            return []

        entries = []
        prefix = definition["prefix"]
        for index, (freq, degree, octave) in enumerate(tones):
            if index == 0:
                next_freq = tones[index + 1][0]
                low = freq / math.sqrt(next_freq / freq)
            else:
                prev_freq = tones[index - 1][0]
                low = math.sqrt(prev_freq * freq)

            if index == len(tones) - 1:
                prev_freq = tones[index - 1][0]
                high = freq * math.sqrt(freq / prev_freq)
            else:
                next_freq = tones[index + 1][0]
                high = math.sqrt(freq * next_freq)

            if high < min_freq or low > max_freq:
                continue

            entry = {
                "note_id": f"{prefix}_{degree:02d}_O{octave:+d}",
                "range_min": max(min_freq, low),
                "range_max": min(max_freq, high),
                "mapped_freq": freq,
            }
            entries.append(entry)

        return entries

    def _get_frequency_mapping_entries(self, mode_key):
        if mode_key == "original_notes":
            return []

        if mode_key in self.frequency_mapping_cache:
            return self.frequency_mapping_cache[mode_key]

        entries = self._build_frequency_mapping_entries(mode_key)
        self.frequency_mapping_cache[mode_key] = entries
        self.frequency_mapping_boundaries[mode_key] = [entry["range_max"] for entry in entries]
        return entries

    def _map_frequency_value(self, frequency, mode_key):
        if frequency <= 0.0:
            return {
                "note_id": "REST",
                "range_min": 0.0,
                "range_max": 0.0,
                "mapped_freq": 0.0,
            }

        if mode_key == "original_notes":
            return {
                "note_id": "ORIGINAL",
                "range_min": frequency,
                "range_max": frequency,
                "mapped_freq": frequency,
            }

        entries = self._get_frequency_mapping_entries(mode_key)
        boundaries = self.frequency_mapping_boundaries.get(mode_key, [])
        if not entries or not boundaries:
            return {
                "note_id": "ORIGINAL",
                "range_min": frequency,
                "range_max": frequency,
                "mapped_freq": frequency,
            }

        entry_index = bisect.bisect_left(boundaries, frequency)
        if entry_index >= len(entries):
            entry_index = len(entries) - 1

        entry = entries[entry_index]
        if frequency < entry["range_min"] and entry_index > 0:
            entry = entries[entry_index - 1]

        return {
            "note_id": entry["note_id"],
            "range_min": entry["range_min"],
            "range_max": entry["range_max"],
            "mapped_freq": entry["mapped_freq"],
        }

    def getFrequencyMappingTable(self):
        mode_key = str(self.params.get("frequency_mapping_mode", "original_notes"))
        if mode_key == "original_notes":
            return []
        return list(self._get_frequency_mapping_entries(mode_key))

    def _waveform_from_mode(self, mode, signed_activation=0.0, activation_spread=0.0):
        if mode == "pure_sine":
            return "sine"
        if mode == "noise_heavy":
            noise_heavy_waves = ["noise", "square", "noise", "triangle", "noise", "sawtooth"]
            self.audio_wave_index = (self.audio_wave_index + 1) % len(noise_heavy_waves)
            return noise_heavy_waves[self.audio_wave_index]
        if mode == "classic_analog":
            classic_waves = ["triangle", "sawtooth", "triangle", "square", "sawtooth"]
            self.audio_wave_index = (self.audio_wave_index + 1) % len(classic_waves)
            return classic_waves[self.audio_wave_index]
        # Neural-reactive waveforms pass through directly — sanasyna reads _neuro_params
        if mode in ("neuro_formant", "neuro_pulse", "neuro_ring", "neuro_fold", "neuro_fm"):
            return mode

        dynamic_offset = int((abs(signed_activation) + activation_spread) * 10)
        self.audio_wave_index = (self.audio_wave_index + 1 + dynamic_offset) % len(self.audio_waveforms)
        return self.audio_waveforms[self.audio_wave_index]

    def _melody_pitch_pool(self, mode_key):
        definition = self._frequency_mapping_definition(mode_key)
        if definition is None:
            return [0.0, 2.0, 4.0, 5.0, 7.0, 9.0, 11.0]

        kind = definition.get("kind")
        if kind == "ratio":
            pool = []
            for ratio in definition.get("ratios", []):
                if ratio <= 0.0:
                    continue
                pool.append(12.0 * math.log2(float(ratio)))
            return pool or [0.0, 2.0, 4.0, 5.0, 7.0, 9.0, 11.0]

        if kind == "et_mode":
            return [float(step) for step in definition.get("steps", [])] or [0.0, 2.0, 4.0, 5.0, 7.0, 9.0, 11.0]

        if kind == "et_full":
            divisions = max(1, int(definition.get("divisions", 12)))
            return [12.0 * (float(step) / float(divisions)) for step in range(divisions)]

        return [0.0, 2.0, 4.0, 5.0, 7.0, 9.0, 11.0]

    def _freq_to_midi(self, frequency):
        if frequency <= 0.0:
            return 0.0
        return 69.0 + 12.0 * math.log2(float(frequency) / 440.0)

    def _midi_to_freq(self, midi_value):
        return 440.0 * (2.0 ** ((float(midi_value) - 69.0) / 12.0))

    def _word_melody_from_gematria(self, word_text, activation_value=0.0):
        letter_values = [gematria_table[letter] for letter in word_text if letter in gematria_table]
        if not letter_values:
            return []

        total = sum(letter_values)
        pattern = []
        max_activation = max(0.001, float(self.params.get("activation_limit", 2.0)))
        activation_norm = min(1.0, max(0.0, abs(float(activation_value)) / max_activation))
        mapping_mode = str(self.params.get("frequency_mapping_mode", "original_notes"))
        melody_coherence = min(1.0, max(0.0, float(self.params.get("melody_coherence", 0.65))))
        pitch_pool = self._melody_pitch_pool(mapping_mode)
        if not pitch_pool:
            pitch_pool = [0.0, 2.0, 4.0, 5.0, 7.0, 9.0, 11.0]

        root_midi = 45.0 + float(total % 12) + float(digital_root(total)) * 0.65
        previous_midi = None
        previous_frequency = 0.0

        for index, value in enumerate(letter_values):
            pool_index = int((value + index + digital_root(total)) % len(pitch_pool))
            octave_span = 1 if melody_coherence >= 0.5 else 2
            octave_shift = ((index + value // max(1, len(pitch_pool))) % (octave_span * 2 + 1)) - octave_span
            target_midi = root_midi + pitch_pool[pool_index] + 12.0 * float(octave_shift)
            root_pull = 0.12 + 0.28 * melody_coherence
            target_midi = target_midi * (1.0 - root_pull) + root_midi * root_pull

            if previous_midi is not None:
                max_leap = 8.0 - 4.5 * melody_coherence
                while target_midi - previous_midi > max_leap:
                    target_midi -= 12.0
                while previous_midi - target_midi > max_leap:
                    target_midi += 12.0
                smooth_mix = 0.45 + 0.35 * melody_coherence
                target_midi = previous_midi * smooth_mix + target_midi * (1.0 - smooth_mix)

            frequency = self._midi_to_freq(target_midi)
            frequency = max(90.0, min(1100.0, frequency))
            mapping_entry = self._map_frequency_value(frequency, mapping_mode)
            frequency = mapping_entry["mapped_freq"]

            if previous_frequency > 0.0:
                melodic_delta = abs(self._freq_to_midi(frequency) - self._freq_to_midi(previous_frequency))
            else:
                melodic_delta = abs(target_midi - root_midi)

            pattern_norm = min(1.0, melodic_delta / 7.0)
            base_duration = 0.07 + (1.0 - pattern_norm) * 0.08
            duration = base_duration * (1.0 - 0.45 * activation_norm)
            duration = min(0.2, max(0.001, duration))
            pattern.append((frequency, duration))
            previous_frequency = frequency
            previous_midi = self._freq_to_midi(frequency)

        reflected = pattern + list(reversed(pattern))
        return reflected

    def _sentence_melody(self):
        melody = []
        for word in self.words:
            melody.extend(self._word_melody_from_gematria(word.word, activation_value=word.neuron.activation))
        return melody

    def _apply_duration_policy(self, melody, speed_coeff=1.0):
        if not melody:
            return melody

        duration_coeff = max(0.05, float(speed_coeff))
        min_note_duration = min(1.0, max(0.01, float(self.params.get("min_note_duration", 0.03))))

        adjusted = []
        positive_durations = []
        for frequency, duration in melody:
            adjusted_duration = max(0.001, float(duration) * duration_coeff)
            adjusted.append((frequency, adjusted_duration))
            if adjusted_duration > 0.0:
                positive_durations.append(adjusted_duration)

        if not positive_durations:
            return adjusted

        current_min_duration = min(positive_durations)
        if current_min_duration >= min_note_duration:
            return adjusted

        scale = min_note_duration / current_min_duration
        return [(frequency, max(0.001, duration * scale)) for frequency, duration in adjusted]

    def updateAudio(self):
        now = time.time()
        if now - self.last_audio_update < self.audio_refresh_interval:
            return
        self.last_audio_update = now

        if len(self.words) == 0:
            if self.audio_playing:
                sanasyna.stop()
                self.audio_playing = False
                self.last_audio_sentence_signature = None
            return

        average_activation = sum(abs(word.neuron.activation) for word in self.words) / len(self.words)
        signed_activation = sum(word.neuron.activation for word in self.words) / len(self.words)
        amplitude = min(0.25, max(0.05, average_activation / 10))

        activation_spread = sum(abs(word.neuron.activation - signed_activation) for word in self.words) / len(self.words)
        mode = self.params.get("audio_wave_mode", "dynamic")

        waveform = self._waveform_from_mode(mode, signed_activation=signed_activation, activation_spread=activation_spread)
        melody = self._sentence_melody()
        if not melody:
            return

        mapping_mode = str(self.params.get("frequency_mapping_mode", "original_notes"))

        voice_count = max(1, min(4, int(self.params.get("voice_count", 1))))
        voice_spread = float(self.params.get("voice_spread", 1.0))
        voice_distance = min(1.0, max(0.0, float(self.params.get("voice_distance", 0.65))))
        voice_distance_context = max(1, int(self.params.get("voice_distance_context", 4)))
        rhythmic_divergence = min(1.0, max(0.0, float(self.params.get("rhythmic_divergence", 0.35))))
        beat_library_style = str(self.params.get("beat_library_style", "auto"))
        rhythm_gate_strength = min(1.0, max(0.0, float(self.params.get("rhythm_gate_strength", 0.85))))
        rhythm_stretch_strength = min(1.0, max(0.0, float(self.params.get("rhythm_stretch_strength", 1.0))))
        rhythm_rotation = max(0, int(self.params.get("rhythm_rotation", 0)))
        rhythm_radicality = min(1.0, max(0.0, float(self.params.get("rhythm_radicality", 0.5))))
        rhythm_style = str(self.params.get("rhythm_style", "manual"))
        strict_counterpoint = bool(self.params.get("strict_counterpoint", False))
        melody_coherence = min(1.0, max(0.0, float(self.params.get("melody_coherence", 0.65))))
        melody_speed = float(self.params.get("melody_speed", 1.0))
        min_note_duration = float(self.params.get("min_note_duration", 0.03))
        effective_voice_spread = max(0.3, min(5.0, voice_spread * (1.15 - 0.55 * melody_coherence)))

        melody = self._apply_duration_policy(melody, speed_coeff=melody_speed)

        melody_from_own_time = bool(self.params.get("melody_from_own_time", True))
        # melody_from_own_time=True  → play once, next period starts when playback ends
        # melody_from_own_time=False → loop until the process_interval timer fires

        if melody_from_own_time:
            # Continue word processing while one-shot audio is playing, but do not
            # start a new synthesis until playback has ended.
            thread_active = self._synthesis_thread is not None and self._synthesis_thread.is_alive()
            if thread_active or sanasyna.is_playing():
                return

        activation_signature = tuple(round(word.neuron.activation, 2) for word in self.words)
        signature = (
            tuple(word.word for word in self.words),
            waveform,
            mapping_mode,
            voice_count,
            strict_counterpoint,
            round(voice_spread, 2),
            round(voice_distance, 2),
            int(voice_distance_context),
            round(rhythmic_divergence, 2),
            beat_library_style,
            round(rhythm_gate_strength, 2),
            round(rhythm_stretch_strength, 2),
            int(rhythm_rotation),
            round(rhythm_radicality, 2),
            rhythm_style,
            round(melody_coherence, 2),
            round(melody_speed, 2),
            round(min_note_duration, 3),
            len(melody),
            melody_from_own_time,
            activation_signature,
        )

        if signature == self.last_audio_sentence_signature and self.audio_playing:
            return

        synth_volume = min(1.0, max(0.0, float(self.params.get("synth_volume", 1.0))))
        melody_amplitude = amplitude * synth_volume
        if waveform == "square":
            melody_amplitude = amplitude * synth_volume * 0.80
        elif waveform == "sawtooth":
            melody_amplitude = amplitude * synth_volume * 0.85
        elif waveform == "triangle":
            melody_amplitude = amplitude * synth_volume * 0.95
        elif waveform == "noise":
            melody_amplitude = max(0.0, amplitude * synth_volume * 0.55)

        # Push current neural activation state so neuro_* waveforms can read it during synthesis
        sanasyna.set_neuro_params({
            "signed_activation": signed_activation,
            "activation_spread": activation_spread,
        })

        # Snapshot all inputs before spawning the thread so later UI changes don't race
        _melody_snap = list(melody)
        _amplitude_snap = melody_amplitude
        _rate_snap = self.audio_sample_rate
        _waveform_snap = waveform
        _vc_snap = voice_count
        _sc_snap = strict_counterpoint
        _evs_snap = effective_voice_spread
        _vd_snap = voice_distance
        _vdc_snap = voice_distance_context
        _rd_snap = rhythmic_divergence
        _bls_snap = beat_library_style
        _rgs_snap = rhythm_gate_strength
        _rss_snap = rhythm_stretch_strength
        _rr_snap = rhythm_rotation
        _rrad_snap = rhythm_radicality
        _mm_snap = mapping_mode
        _loop_snap = not melody_from_own_time  # True=loop (cut by timer), False=play once

        # If a synthesis is already running it will land via _pending_samples / crossfade
        if self._synthesis_thread is not None and self._synthesis_thread.is_alive():
            return

        # Mark signature only when a new synthesis will actually be started.
        # If we mark it while a thread is still running, future ticks can falsely
        # believe synthesis already happened and leave old loops playing too long.
        self.last_audio_sentence_signature = signature
        self.audio_playing = True

        def _run_synthesis():
            sanasyna.generate_melody(
                _melody_snap,
                _amplitude_snap,
                _rate_snap,
                duration_per_note=0.09,
                waveform=_waveform_snap,
                voices=_vc_snap,
                counterpoint=True,
                strict_counterpoint=_sc_snap,
                voice_spread=_evs_snap,
                voice_distance=_vd_snap,
                voice_distance_context=_vdc_snap,
                rhythmic_divergence=_rd_snap,
                beat_library_style=_bls_snap,
                rhythm_gate_strength=_rgs_snap,
                rhythm_stretch_strength=_rss_snap,
                rhythm_rotation=_rr_snap,
                rhythm_radicality=_rrad_snap,
                mapping_mode=_mm_snap,
                duration_coeff=1.0,
            )
            sanasyna.play(loop=_loop_snap)

        self._synthesis_thread = threading.Thread(target=_run_synthesis, daemon=True)
        self._synthesis_thread.start()

    def makeWordCircle(self, words):
        zoom = self.params["zoom"]
        for i, word in enumerate(words):
            word.x = self.size[0]/2 + 6*zoom * math.cos(2 * math.pi * i / len(words))
            word.y = self.size[1]/2 + 6*zoom * math.sin(2 * math.pi * i / len(words))
            word.neuron.x = word.x
            word.neuron.y = word.y


    def initWords(self):
        input_filename = sys.argv[1] if len(sys.argv) > 1 else None
        reference_filename = sys.argv[2] if len(sys.argv) > 2 else None

        self.words = self.parseText(input_filename) if input_filename else []
        self.referenceWords = self.parseText(reference_filename) if reference_filename else []

        for word in self.words:
            self._assign_seed_pos(word)
            self.referenceWords.append(word)

        self.referenceWords = self._uniqueWordObjects(self.referenceWords)
        self._markReferenceIndexDirty()

        #place words in a circle
        self.makeWordCircle(self.words)

        #connect words
        for word in self.words:
            for word2 in self.words:
                #prevent self-connection and eternal recursion
                if word != word2:
                    weight = self.getGematriaDistance(word.gematria, word2.gematria)
                    #weight += get_distance(word.x, word.y, word2.x, word2.y) / 1000
                    word.connect(word2, weight)
                    print (word.word + " connected to " + word2.word + " with weight " + str(weight))

    def parseText(self, filename):
        words = []
        valid_chars = gematria_table.keys()
        with open(filename, "r") as file:
            for line in file:
                for word in line.split():
                    word = word.lower()
                    if all(char in valid_chars for char in word):
                        words.append(Word(word, 0, 0, (255, 255, 255), self))
        return words

    def _uniqueWordObjects(self, words):
        unique_words = []
        seen_words = set()
        for word in words:
            if word.word not in seen_words:
                unique_words.append(word)
                seen_words.add(word.word)
        return unique_words

    def _markReferenceIndexDirty(self):
        self.reference_index_dirty = True

    def _rebuildReferenceIndex(self, include_pos=False):
        by_gematria = {}
        by_reduction = {}
        by_root = {}
        by_pos_gematria = {}
        by_pos_reduction = {}
        by_pos_root = {}

        for ref_word in self.referenceWords:
            gematria_value = ref_word.gematria
            reduction_value = numerological_reduction(gematria_value)
            root_value = digital_root(gematria_value)

            by_gematria.setdefault(gematria_value, []).append(ref_word)
            by_reduction.setdefault(reduction_value, []).append(ref_word)
            by_root.setdefault(root_value, []).append(ref_word)

            if include_pos:
                pos_value = self.getWordPOS(ref_word.word, force=True)
                by_pos_gematria.setdefault((pos_value, gematria_value), []).append(ref_word)
                by_pos_reduction.setdefault((pos_value, reduction_value), []).append(ref_word)
                by_pos_root.setdefault((pos_value, root_value), []).append(ref_word)

        self.reference_index = {
            "gematria": by_gematria,
            "reduction": by_reduction,
            "root": by_root,
            "pos_gematria": by_pos_gematria,
            "pos_reduction": by_pos_reduction,
            "pos_root": by_pos_root,
        }
        self.reference_index_dirty = False
        self.reference_index_has_pos = include_pos

    def _ensureReferenceIndex(self, include_pos=False):
        if self.reference_index_dirty:
            self._rebuildReferenceIndex(include_pos=include_pos)
            return

        if include_pos and not self.reference_index_has_pos:
            self._rebuildReferenceIndex(include_pos=True)

    def _ltm_candidate_probabilities(self, source_word, candidates):
        if not self._is_ltm_enabled():
            return {}
        if self.ltm_model is None:
            return {}
        if not candidates:
            return {}

        context_words = [word.word for word in self.words if hasattr(word, "word") and isinstance(word.word, str) and word.word]
        if source_word is not None and isinstance(source_word.word, str) and source_word.word:
            if not context_words or context_words[-1] != source_word.word:
                context_words.append(source_word.word)

        model_context_size = max(1, int(getattr(self.ltm_model, "context_size", self.ltm_context_size)))
        context_words = context_words[-model_context_size:]
        candidate_words = list({candidate.word for candidate in candidates if isinstance(candidate.word, str) and candidate.word})
        if not candidate_words:
            return {}

        try:
            probabilities = self.ltm_model.predict_next_probabilities(context_words=context_words, candidate_words=candidate_words)
            if isinstance(probabilities, dict):
                return probabilities
        except Exception:
            return {}
        return {}

    def _common_word_penalty_factor(self, word_text):
        if not bool(self.params.get("common_word_penalty", True)):
            return 1.0
        if not isinstance(word_text, str):
            return 1.0

        common_words = {
            "the", "and", "is", "a", "an", "to", "of", "in", "on", "for", "with", "at", "by", "from",
            "it", "this", "that", "these", "those", "as", "be", "are", "was", "were", "am", "i", "you",
            "he", "she", "we", "they", "me", "him", "her", "them", "my", "your", "our", "their",
        }
        return 0.35 if word_text.lower() in common_words else 1.0

    def graphemeToPhonemes(self, word_text):
        if not isinstance(word_text, str):
            return ()

        lowered_word = word_text.lower().strip()
        if lowered_word == "":
            return ()

        cached = self.phoneme_cache.get(lowered_word)
        if cached is not None:
            return cached

        normalized = "".join(char for char in lowered_word if char in gematria_table)
        if normalized == "":
            self.phoneme_cache[lowered_word] = ()
            return ()

        whole_word_overrides = {
            "you": ("u",),
            "your": ("y", "or"),
            "yours": ("y", "or", "z"),
            "one": ("w", "ah", "n"),
            "once": ("w", "ah", "n", "s"),
        }
        override = whole_word_overrides.get(normalized)
        if override is not None:
            self.phoneme_cache[lowered_word] = override
            return override

        phonemes = []
        index = 0

        def _is_vowel(char_value):
            return char_value in {"a", "e", "i", "o", "u", "y", "å", "ä", "ö"}

        while index < len(normalized):
            remaining = normalized[index:]

            if remaining.startswith("tion"):
                phonemes.extend(["sh", "un"])
                index += 4
                continue
            if remaining.startswith("sion"):
                phonemes.extend(["zh", "un"])
                index += 4
                continue
            if remaining.startswith("ture"):
                phonemes.extend(["ch", "er"])
                index += 4
                continue

            if remaining.startswith("dge"):
                phonemes.append("j")
                index += 3
                continue
            if remaining.startswith("tch"):
                phonemes.append("ch")
                index += 3
                continue
            if remaining.startswith("igh"):
                phonemes.append("ai")
                index += 3
                continue

            pair = normalized[index:index + 2]
            if pair == "ng":
                phonemes.append("ng")
                index += 2
                continue
            if pair == "nk":
                phonemes.extend(["ng", "k"])
                index += 2
                continue
            if pair == "ph":
                phonemes.append("f")
                index += 2
                continue
            if pair == "sh":
                phonemes.append("sh")
                index += 2
                continue
            if pair == "ch":
                phonemes.append("ch")
                index += 2
                continue
            if pair == "th":
                phonemes.append("th")
                index += 2
                continue
            if pair == "wh":
                phonemes.append("w")
                index += 2
                continue
            if pair == "qu":
                phonemes.extend(["k", "w"])
                index += 2
                continue
            if pair == "ck":
                phonemes.append("k")
                index += 2
                continue
            if pair == "ee":
                phonemes.append("i")
                index += 2
                continue
            if pair == "ea":
                phonemes.append("i")
                index += 2
                continue
            if pair == "oo":
                phonemes.append("u")
                index += 2
                continue
            if pair == "oa":
                phonemes.append("ou")
                index += 2
                continue
            if pair == "ai":
                phonemes.append("ei")
                index += 2
                continue
            if pair == "ay":
                phonemes.append("ei")
                index += 2
                continue
            if pair == "oy":
                phonemes.append("oi")
                index += 2
                continue
            if pair == "oi":
                phonemes.append("oi")
                index += 2
                continue
            if pair == "ou":
                phonemes.append("au")
                index += 2
                continue
            if pair == "ow":
                if index + 2 == len(normalized):
                    phonemes.append("au")
                else:
                    phonemes.append("ou")
                index += 2
                continue
            if pair == "au":
                phonemes.append("o")
                index += 2
                continue
            if pair == "er":
                phonemes.append("er")
                index += 2
                continue
            if pair == "ar":
                phonemes.append("ar")
                index += 2
                continue
            if pair == "or":
                phonemes.append("or")
                index += 2
                continue

            char = normalized[index]
            next_char = normalized[index + 1] if index + 1 < len(normalized) else ""
            prev_char = normalized[index - 1] if index > 0 else ""

            if char == "c":
                if next_char in {"e", "i", "y"}:
                    phonemes.append("s")
                else:
                    phonemes.append("k")
            elif char == "g":
                if next_char in {"e", "i", "y"}:
                    phonemes.append("j")
                else:
                    phonemes.append("g")
            elif char == "q":
                phonemes.append("k")
            elif char == "w":
                phonemes.append("w")
            elif char == "x":
                phonemes.extend(["k", "s"])
            elif char == "a":
                phonemes.append("ae")
            elif char == "e":
                is_final_silent_e = (
                    index == len(normalized) - 1
                    and len(normalized) > 2
                    and not _is_vowel(prev_char)
                )
                if not is_final_silent_e:
                    phonemes.append("e")
            elif char == "i":
                phonemes.append("i")
            elif char == "o":
                phonemes.append("o")
            elif char == "u":
                phonemes.append("u")
            elif char == "y":
                is_vowel_like = index == len(normalized) - 1 or (
                    not _is_vowel(prev_char) and (next_char == "" or not _is_vowel(next_char))
                )
                phonemes.append("i" if is_vowel_like else "y")
            elif char == "å":
                phonemes.append("o")
            elif char == "ä":
                phonemes.append("ae")
            elif char == "ö":
                phonemes.append("oe")
            else:
                phonemes.append(char)
            index += 1

        phonemes_tuple = tuple(phonemes)
        self.phoneme_cache[lowered_word] = phonemes_tuple
        return phonemes_tuple

    def _phoneme_rhyme_tail(self, phonemes):
        if not phonemes:
            return ()

        vowel_phones = {
            "a", "e", "i", "o", "u", "y", "ae", "oe",
            "ah", "aa", "ai", "ei", "oi", "au", "ou",
            "er", "ar", "or", "ur", "ir", "un",
        }
        for index in range(len(phonemes) - 1, -1, -1):
            if phonemes[index] in vowel_phones:
                return phonemes[index:]

        return phonemes

    def _phoneme_suffix_similarity(self, source_phonemes, candidate_phonemes):
        source_tail = self._phoneme_rhyme_tail(source_phonemes)
        candidate_tail = self._phoneme_rhyme_tail(candidate_phonemes)
        if not source_tail or not candidate_tail:
            return 0.0

        overlap = min(len(source_tail), len(candidate_tail))
        matching_suffix = 0
        for offset in range(1, overlap + 1):
            if source_tail[-offset] == candidate_tail[-offset]:
                matching_suffix += 1
            else:
                break

        if matching_suffix == 0:
            return 0.0

        similarity = matching_suffix / float(max(len(source_tail), len(candidate_tail)))
        if source_tail == candidate_tail:
            similarity = min(1.0, similarity + 0.15)
        return similarity

    def _phoneme_full_similarity(self, source_phonemes, candidate_phonemes):
        if not source_phonemes or not candidate_phonemes:
            return 0.0

        n = len(source_phonemes)
        m = len(candidate_phonemes)
        if n == 0 or m == 0:
            return 0.0

        prev = list(range(m + 1))
        for i in range(1, n + 1):
            curr = [i] + [0] * m
            src = source_phonemes[i - 1]
            for j in range(1, m + 1):
                cost = 0 if src == candidate_phonemes[j - 1] else 1
                curr[j] = min(
                    prev[j] + 1,
                    curr[j - 1] + 1,
                    prev[j - 1] + cost,
                )
            prev = curr

        distance = prev[m]
        max_len = max(n, m)
        if max_len == 0:
            return 0.0
        return max(0.0, 1.0 - (distance / float(max_len)))

    def phonemeRhymeSimilarity(self, source_word_text, candidate_word_text):
        source_phonemes = self.graphemeToPhonemes(source_word_text)
        candidate_phonemes = self.graphemeToPhonemes(candidate_word_text)
        if not source_phonemes or not candidate_phonemes:
            return 0.0

        suffix_similarity = self._phoneme_suffix_similarity(source_phonemes, candidate_phonemes)
        full_similarity = self._phoneme_full_similarity(source_phonemes, candidate_phonemes)

        strategy = str(self.params.get("rhyme_strategy", "hybrid"))
        if strategy == "suffix":
            return suffix_similarity
        if strategy == "whole_word":
            return full_similarity

        tail_bias = min(1.0, max(0.0, float(self.params.get("rhyme_tail_bias", 0.60))))
        return tail_bias * suffix_similarity + (1.0 - tail_bias) * full_similarity

    def _reference_blended_score(self, source_word, candidate, ltm_probabilities):
        source_gematria = source_word.gematria
        source_reduction = numerological_reduction(source_gematria)
        source_root = digital_root(source_gematria)

        gematria_delta = abs(candidate.gematria - source_gematria) / 1000.0
        reduction_delta = abs(numerological_reduction(candidate.gematria) - source_reduction) / 9.0
        root_delta = abs(digital_root(candidate.gematria) - source_root) / 9.0
        base_score = gematria_delta + 0.35 * reduction_delta + 0.35 * root_delta

        if ltm_probabilities:
            raw_ltm_prob = float(ltm_probabilities.get(candidate.word, 0.0))
            adjusted_ltm_prob = raw_ltm_prob * self._common_word_penalty_factor(candidate.word)
            ltm_term = 1.0 - adjusted_ltm_prob
            ltm_weight = min(1.0, max(0.0, float(self.params.get("ltm_weight", 0.35))))
        else:
            ltm_term = 0.0
            ltm_weight = 0.0

        rhyme_bonus = 0.0
        if bool(self.params.get("use_phoneme_rhyme", True)) and candidate.word != source_word.word:
            rhyme_similarity = self.phonemeRhymeSimilarity(source_word.word, candidate.word)
            min_similarity = min(1.0, max(0.0, float(self.params.get("rhyme_min_similarity", 0.34))))
            if rhyme_similarity >= min_similarity:
                rhyme_bonus = rhyme_similarity

        rhyme_weight = min(1.0, max(0.0, float(self.params.get("rhyme_weight", 0.28))))

        blended_score = (1.0 - ltm_weight) * base_score + ltm_weight * ltm_term - rhyme_weight * rhyme_bonus
        if blended_score < 0.0:
            blended_score = 0.0
        return blended_score, base_score

    def _selectBestReference(self, source_word, candidates, force_jump=False):
        if not candidates:
            return None

        source_gematria = source_word.gematria
        source_reduction = numerological_reduction(source_gematria)
        source_root = digital_root(source_gematria)
        ltm_probabilities = self._ltm_candidate_probabilities(source_word, candidates)

        scored_candidates = []
        for candidate in candidates:
            blended_score, base_score = self._reference_blended_score(source_word, candidate, ltm_probabilities)
            scored_candidates.append(
                (
                    blended_score,
                    base_score,
                    abs(candidate.gematria - source_gematria),
                    abs(numerological_reduction(candidate.gematria) - source_reduction),
                    abs(digital_root(candidate.gematria) - source_root),
                    candidate.word,
                    candidate,
                )
            )

        scored_candidates.sort(key=lambda item: item[:6])
        ranked_candidates = [item[6] for item in scored_candidates]
        alternative_ranked_candidates = [candidate for candidate in ranked_candidates if candidate.word != source_word.word]

        exploration = min(1.0, max(0.0, float(self.params.get("selection_exploration", 0.18))))
        top_k = max(1, int(self.params.get("selection_top_k", 4)))

        if force_jump and len(ranked_candidates) > 1:
            jump_span = min(len(ranked_candidates), max(6, top_k * 3))
            jump_pool = [candidate for candidate in ranked_candidates[1:jump_span] if candidate.word != source_word.word]
            if not jump_pool:
                jump_pool = [candidate for candidate in ranked_candidates if candidate.word != source_word.word]
            if jump_pool:
                return random.choice(jump_pool)

        if len(ranked_candidates) > 1 and random.random() < exploration:
            top_candidates = ranked_candidates[:min(top_k, len(ranked_candidates))]
            if alternative_ranked_candidates:
                top_candidates = [candidate for candidate in top_candidates if candidate.word != source_word.word]
                if not top_candidates:
                    top_candidates = alternative_ranked_candidates[:min(top_k, len(alternative_ranked_candidates))]
            weights = [1.0 / float(index + 1) for index in range(len(top_candidates))]
            return random.choices(top_candidates, weights=weights, k=1)[0]

        if alternative_ranked_candidates and ranked_candidates[0].word == source_word.word:
            return alternative_ranked_candidates[0]

        return ranked_candidates[0]

    def importReferenceDatabase(self, filename, mode="append"):
        imported_words = self.parseText(filename)
        if not imported_words:
            return 0, len(self.referenceWords)

        if mode == "replace":
            self.referenceWords = self._uniqueWordObjects(imported_words)
        else:
            self.referenceWords = self._uniqueWordObjects(self.referenceWords + imported_words)

        self.referenceWords = self._uniqueWordObjects(self.referenceWords + self.words)
        self._markReferenceIndexDirty()
        return len(imported_words), len(self.referenceWords)

    def OnImportDatabaseFile(self, event):
        with wx.FileDialog(
            self.frame,
            "Import database text file",
            wildcard="Text files (*.txt)|*.txt|All files (*.*)|*.*",
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
        ) as file_dialog:
            if file_dialog.ShowModal() == wx.ID_CANCEL:
                return

            selected_path = file_dialog.GetPath()

        try:
            import_mode = self.params.get("import_mode", "append")
            imported_count, total_count = self.importReferenceDatabase(selected_path, mode=import_mode)
            if imported_count == 0:
                self.import_db_status.SetLabel("No valid words found in selected file")
            else:
                self.import_db_status.SetLabel(f"Imported {imported_count} words ({import_mode}), database size {total_count}")
        except Exception as error:
            self.import_db_status.SetLabel(f"Import failed: {error}")

    def _normalize_params(self):
        self.params["set_weight_by_gematria"] = bool(self.params.get("set_weight_by_gematria", False))
        self.params["use_pos_matching"] = bool(self.params.get("use_pos_matching", False))
        self.params["fluid_pos"] = bool(self.params.get("fluid_pos", False))
        self.params["use_long_term_memory"] = bool(self.params.get("use_long_term_memory", False))
        self.params["common_word_penalty"] = bool(self.params.get("common_word_penalty", True))
        self.params["fluid_root"] = bool(self.params.get("fluid_root", False))
        self.params["fluid_gematria"] = bool(self.params.get("fluid_gematria", False))
        self.params["use_phoneme_rhyme"] = bool(self.params.get("use_phoneme_rhyme", True))
        self.params["strict_counterpoint"] = bool(self.params.get("strict_counterpoint", False))
        self.params["piper_tts_on"] = bool(self.params.get("piper_tts_on", False))
        self.params["process_interval_from_rhythm_bpm"] = bool(self.params.get("process_interval_from_rhythm_bpm", False))
        self.params["compressor_enabled"] = bool(self.params.get("compressor_enabled", False))
        self.params["ltm_weight"] = min(1.0, max(0.0, float(self.params.get("ltm_weight", 0.35))))
        self.params["rhyme_weight"] = min(1.0, max(0.0, float(self.params.get("rhyme_weight", 0.28))))
        self.params["rhyme_min_similarity"] = min(1.0, max(0.0, float(self.params.get("rhyme_min_similarity", 0.34))))
        self.params["piper_model_path"] = str(self.params.get("piper_model_path", "")).strip()
        self.params["compressor_threshold_db"] = min(0.0, max(-48.0, float(self.params.get("compressor_threshold_db", -18.0))))
        self.params["compressor_ratio"] = min(12.0, max(1.0, float(self.params.get("compressor_ratio", 3.0))))
        self.params["compressor_makeup_db"] = min(24.0, max(0.0, float(self.params.get("compressor_makeup_db", 6.0))))

        self.params["learning_rate"] = float(self.params.get("learning_rate", 0.1))
        self.params["error"] = float(self.params.get("error", 0))
        self.params["target"] = float(self.params.get("target", 0))
        self.params["activation_increase"] = float(self.params.get("activation_increase", 0.0001))
        self.params["activation_limit"] = float(self.params.get("activation_limit", 2))
        self.params["sigmoid_scale"] = float(self.params.get("sigmoid_scale", 2))
        self.params["word_change_threshold"] = float(self.params.get("word_change_threshold", 0.777))
        self.params["zoom"] = max(0.001, float(self.params.get("zoom", 0.1)))
        self.params["process_interval"] = min(8.0, max(0.1, float(self.params.get("process_interval", 0.1))))
        self.params["melody_from_own_time"] = bool(self.params.get("melody_from_own_time", True))
        self.params["logic_iteration_limit"] = max(1, int(float(self.params.get("logic_iteration_limit", 48))))
        self.params["selection_exploration"] = min(1.0, max(0.0, float(self.params.get("selection_exploration", 0.18))))
        self.params["selection_top_k"] = max(1, int(float(self.params.get("selection_top_k", 4))))
        self.params["jump_probability"] = min(1.0, max(0.0, float(self.params.get("jump_probability", 0.08))))
        self.params["jump_radius"] = max(0, int(float(self.params.get("jump_radius", 120))))

        import_mode = str(self.params.get("import_mode", "append"))
        if import_mode not in {"append", "replace"}:
            import_mode = "append"
        self.params["import_mode"] = import_mode

        audio_wave_mode = str(self.params.get("audio_wave_mode", "dynamic"))
        if audio_wave_mode not in {
            "dynamic", "pure_sine", "noise_heavy", "classic_analog",
            "neuro_formant", "neuro_pulse", "neuro_ring", "neuro_fold", "neuro_fm",
        }:
            audio_wave_mode = "dynamic"
        self.params["audio_wave_mode"] = audio_wave_mode

        frequency_mapping_mode = str(self.params.get("frequency_mapping_mode", "original_notes"))
        valid_frequency_modes = {key for key, _ in self._frequency_mapping_modes()}
        if frequency_mapping_mode not in valid_frequency_modes:
            frequency_mapping_mode = "original_notes"
        self.params["frequency_mapping_mode"] = frequency_mapping_mode

        rhythm_style = str(self.params.get("rhythm_style", "manual"))
        valid_rhythm_styles = {key for key, _ in self._rhythm_style_modes()}
        if rhythm_style not in valid_rhythm_styles:
            rhythm_style = "manual"
        self.params["rhythm_style"] = rhythm_style

        beat_library_style = str(self.params.get("beat_library_style", "auto"))
        valid_beat_styles = {key for key, _ in self._beat_library_modes()}
        if beat_library_style not in valid_beat_styles:
            beat_library_style = "auto"
        self.params["beat_library_style"] = beat_library_style

        self.params["voice_count"] = max(1, min(4, int(float(self.params.get("voice_count", 1)))))
        self.params["voice_spread"] = min(5.0, max(0.3, float(self.params.get("voice_spread", 1.0))))
        self.params["voice_distance"] = min(1.0, max(0.0, float(self.params.get("voice_distance", 0.65))))
        self.params["voice_distance_context"] = max(1, min(32, int(float(self.params.get("voice_distance_context", 4)))))
        self.params["rhythmic_divergence"] = min(1.0, max(0.0, float(self.params.get("rhythmic_divergence", 0.35))))
        self.params["rhythm_gate_strength"] = min(1.0, max(0.0, float(self.params.get("rhythm_gate_strength", 0.85))))
        self.params["rhythm_stretch_strength"] = min(1.0, max(0.0, float(self.params.get("rhythm_stretch_strength", 1.0))))
        self.params["rhythm_rotation"] = max(0, min(31, int(float(self.params.get("rhythm_rotation", 0)))))
        self.params["rhythm_radicality"] = min(1.0, max(0.0, float(self.params.get("rhythm_radicality", 0.5))))
        self.params["rhythm_mod_bpm"] = min(300.0, max(0.01, float(self.params.get("rhythm_mod_bpm", 108.0))))
        if self.params["process_interval_from_rhythm_bpm"]:
            self.params["process_interval"] = self._process_interval_from_bpm(self.params["rhythm_mod_bpm"])
        additive_blocks = self.params.get("additive_rhythm_blocks", [])
        normalized_blocks = []
        if isinstance(additive_blocks, (list, tuple)):
            for value in additive_blocks:
                try:
                    block = int(float(value))
                except Exception:
                    continue
                if block > 0:
                    normalized_blocks.append(block)
        self.params["additive_rhythm_blocks"] = normalized_blocks
        self.params["additive_rhythm_weight"] = min(1.0, max(0.0, float(self.params.get("additive_rhythm_weight", 0.0))))
        divisive_signature = str(self.params.get("divisive_rhythm_signature", "4/4"))
        valid_signatures = {key for key, _ in self._divisive_signature_modes()}
        if divisive_signature not in valid_signatures:
            divisive_signature = "4/4"
        self.params["divisive_rhythm_signature"] = divisive_signature
        self.params["divisive_rhythm_weight"] = min(1.0, max(0.0, float(self.params.get("divisive_rhythm_weight", 0.0))))
        self.params["melody_coherence"] = min(1.0, max(0.0, float(self.params.get("melody_coherence", 0.65))))
        self.params["melody_speed"] = min(6.0, max(0.2, float(self.params.get("melody_speed", 1.0))))
        self.params["min_note_duration"] = min(1.0, max(0.01, float(self.params.get("min_note_duration", 0.03))))

        self.params["adsr_attack"] = max(0.0, float(self.params.get("adsr_attack", 0.01)))
        self.params["adsr_decay"] = max(0.0, float(self.params.get("adsr_decay", 0.04)))
        self.params["adsr_sustain"] = min(1.0, max(0.0, float(self.params.get("adsr_sustain", 0.85))))
        self.params["adsr_release"] = max(0.0, float(self.params.get("adsr_release", 0.03)))

        rhyme_strategy = str(self.params.get("rhyme_strategy", "hybrid"))
        if rhyme_strategy not in {"suffix", "whole_word", "hybrid"}:
            rhyme_strategy = "hybrid"
        self.params["rhyme_strategy"] = rhyme_strategy
        self.params["rhyme_tail_bias"] = min(1.0, max(0.0, float(self.params.get("rhyme_tail_bias", 0.60))))

    def _sync_controls_from_params(self):
        self._suppress_param_events = True
        try:
            self.set_weight_by_gematria_checkbox.SetValue(self.params["set_weight_by_gematria"])
            self.use_pos_matching_checkbox.SetValue(self.params["use_pos_matching"])
            self.fluid_pos_checkbox.SetValue(self.params["fluid_pos"])
            self.use_long_term_memory_checkbox.SetValue(self.params["use_long_term_memory"])
            self.common_word_penalty_checkbox.SetValue(self.params["common_word_penalty"])
            self.fluid_root_checkbox.SetValue(self.params["fluid_root"])
            self.fluid_gematria_checkbox.SetValue(self.params["fluid_gematria"])
            self.use_phoneme_rhyme_checkbox.SetValue(self.params["use_phoneme_rhyme"])
            self.strict_counterpoint_checkbox.SetValue(self.params["strict_counterpoint"])
            self.melody_from_own_time_checkbox.SetValue(self.params.get("melody_from_own_time", True))
            self.process_interval_bpm_sync_checkbox.SetValue(bool(self.params.get("process_interval_from_rhythm_bpm", False)))
            self.piper_tts_checkbox.SetValue(self.params.get("piper_tts_on", False))
            self.compressor_enabled_checkbox.SetValue(bool(self.params.get("compressor_enabled", False)))
            self.fullscreen_checkbox.SetValue(bool(self.params.get("fullscreen", False)))
            self._updatePOSBackendStatusLabel(check_nltk=False)
            self._updateLTMStatusLabel()
            self._setCtrlValueSilently(self.ltm_weight_ctrl, self.params["ltm_weight"])
            self._setCtrlValueSilently(self.rhyme_weight_ctrl, self.params["rhyme_weight"])
            self._setCtrlValueSilently(self.rhyme_min_similarity_ctrl, self.params["rhyme_min_similarity"])
            self.rhyme_strategy_choice.SetStringSelection(self._rhyme_strategy_label_from_key(self.params.get("rhyme_strategy", "hybrid")))
            self._setCtrlValueSilently(self.rhyme_tail_bias_ctrl, self.params.get("rhyme_tail_bias", 0.60))
            self._setCtrlValueSilently(self.piper_model_ctrl, self.params.get("piper_model_path", ""))

            self._setCtrlValueSilently(self.learning_rate_ctrl, self.params["learning_rate"])
            self._setCtrlValueSilently(self.error_ctrl, self.params["error"])
            self._setCtrlValueSilently(self.activation_increase_ctrl, self.params["activation_increase"])
            self._setCtrlValueSilently(self.activation_limit_ctrl, self.params["activation_limit"])
            self._setCtrlValueSilently(self.sigmoid_scale_ctrl, self.params["sigmoid_scale"])
            self._setCtrlValueSilently(self.word_change_threshold_ctrl, self.params["word_change_threshold"])
            self._setCtrlValueSilently(self.zoom_ctrl, self.params["zoom"])
            self._setCtrlValueSilently(self.process_interval_ctrl, self.params["process_interval"])
            self._update_process_interval_mode_ui()
            self._setCtrlValueSilently(self.selection_exploration_ctrl, self.params["selection_exploration"])
            self._setCtrlValueSilently(self.selection_top_k_ctrl, self.params["selection_top_k"])
            self._setCtrlValueSilently(self.jump_probability_ctrl, self.params["jump_probability"])
            self._setCtrlValueSilently(self.jump_radius_ctrl, self.params["jump_radius"])
            self.compressor_threshold_slider.SetValue(int(round(self.params.get("compressor_threshold_db", -18.0))))
            self.compressor_ratio_slider.SetValue(int(round(float(self.params.get("compressor_ratio", 3.0)) * 10.0)))
            self.compressor_makeup_slider.SetValue(int(round(self.params.get("compressor_makeup_db", 6.0))))
            self._update_compressor_labels()

            _wave_mode_to_label = {
                "pure_sine": "Pure sine",
                "noise_heavy": "Noise-heavy",
                "classic_analog": "Classic analog",
                "neuro_formant": "Neuro formant",
                "neuro_pulse": "Neuro pulse",
                "neuro_ring": "Neuro ring",
                "neuro_fold": "Neuro fold",
                "neuro_fm": "Neuro FM",
            }
            self.audio_wave_mode_choice.SetStringSelection(
                _wave_mode_to_label.get(self.params["audio_wave_mode"], "Dynamic")
            )
            self.frequency_mapping_choice.SetStringSelection(self._frequency_mapping_label_from_key(self.params["frequency_mapping_mode"]))
            self._apply_fullscreen_mode(bool(self.params.get("fullscreen", False)))
            self.rhythm_style_choice.SetStringSelection(self._rhythm_style_label_from_key(self.params.get("rhythm_style", "manual")))
            self.beat_library_style_choice.SetStringSelection(self._beat_library_label_from_key(self.params.get("beat_library_style", "auto")))
            self.voice_count_choice.SetStringSelection(str(self.params["voice_count"]))
            self._setCtrlValueSilently(self.voice_spread_ctrl, self.params["voice_spread"])
            self._setCtrlValueSilently(self.voice_distance_ctrl, self.params["voice_distance"])
            self._setCtrlValueSilently(self.voice_distance_context_ctrl, self.params["voice_distance_context"])
            self._setCtrlValueSilently(self.rhythmic_divergence_ctrl, self.params["rhythmic_divergence"])
            self._setCtrlValueSilently(self.rhythm_gate_strength_ctrl, self.params["rhythm_gate_strength"])
            self._setCtrlValueSilently(self.rhythm_stretch_strength_ctrl, self.params["rhythm_stretch_strength"])
            self._setCtrlValueSilently(self.rhythm_rotation_ctrl, self.params["rhythm_rotation"])
            self._setCtrlValueSilently(self.rhythm_radicality_ctrl, self.params["rhythm_radicality"])
            self._setCtrlValueSilently(self.rhythm_mod_bpm_ctrl, self.params.get("rhythm_mod_bpm", 108.0))
            self.divisive_signature_choice.SetStringSelection(self._divisive_signature_label_from_key(self.params.get("divisive_rhythm_signature", "4/4")))
            add_weight_pct = int(min(100, max(0, round(self.params.get("additive_rhythm_weight", 0.0) * 100))))
            div_weight_pct = int(min(100, max(0, round(self.params.get("divisive_rhythm_weight", 0.0) * 100))))
            self.additive_rhythm_weight_slider.SetValue(add_weight_pct)
            self.divisive_rhythm_weight_slider.SetValue(div_weight_pct)
            self.additive_rhythm_weight_label.SetLabel(f"Additive rhythm weight: {add_weight_pct}%")
            self.divisive_rhythm_weight_label.SetLabel(f"Divisive rhythm weight: {div_weight_pct}%")
            self._setCtrlValueSilently(self.melody_coherence_ctrl, self.params["melody_coherence"])
            self._setCtrlValueSilently(self.melody_speed_ctrl, self.params["melody_speed"])
            self._setCtrlValueSilently(self.min_note_duration_ctrl, self.params["min_note_duration"])

            if self.params["import_mode"] == "replace":
                self.import_mode_choice.SetStringSelection("Replace database")
            else:
                self.import_mode_choice.SetStringSelection("Append database")

            self._setCtrlValueSilently(self.adsr_attack_ctrl, self.params["adsr_attack"])
            self._setCtrlValueSilently(self.adsr_decay_ctrl, self.params["adsr_decay"])
            self._setCtrlValueSilently(self.adsr_sustain_ctrl, self.params["adsr_sustain"])
            self._setCtrlValueSilently(self.adsr_release_ctrl, self.params["adsr_release"])
            self._update_additive_pattern_status_label()
            self._apply_rhythm_modulation_state()
            self._apply_compressor_state()
            self._refresh_additive_editor_sequence()
        finally:
            self._suppress_param_events = False

    def _apply_loaded_preset(self):
        self._normalize_params()
        self._sync_controls_from_params()
        self._applyADSRToAudio()
        self.makeWordCircle(self.words)
        self.last_audio_sentence_signature = None

    def OnSavePreset(self, event):
        with wx.FileDialog(
            self.frame,
            "Save preset JSON",
            wildcard="JSON files (*.json)|*.json|All files (*.*)|*.*",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
            defaultFile="sanaverkko_preset.json",
        ) as file_dialog:
            if file_dialog.ShowModal() == wx.ID_CANCEL:
                return

            selected_path = file_dialog.GetPath()

        try:
            self._normalize_params()
            with open(selected_path, "w", encoding="utf-8") as outfile:
                json.dump(self.params, outfile, ensure_ascii=False, indent=2)
            self.preset_status.SetLabel(f"Preset saved: {os.path.basename(selected_path)}")
        except Exception as error:
            self.preset_status.SetLabel(f"Preset save failed: {error}")

    def OnLoadPreset(self, event):
        with wx.FileDialog(
            self.frame,
            "Load preset JSON",
            wildcard="JSON files (*.json)|*.json|All files (*.*)|*.*",
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
        ) as file_dialog:
            if file_dialog.ShowModal() == wx.ID_CANCEL:
                return

            selected_path = file_dialog.GetPath()

        try:
            with open(selected_path, "r", encoding="utf-8") as infile:
                loaded_data = json.load(infile)

            if not isinstance(loaded_data, dict):
                raise ValueError("Preset JSON must be an object")

            for key, value in loaded_data.items():
                if key in self.params:
                    self.params[key] = value

            self._apply_loaded_preset()
            self.preset_status.SetLabel(f"Preset loaded: {os.path.basename(selected_path)}")
        except Exception as error:
            self.preset_status.SetLabel(f"Preset load failed: {error}")

    def addWordToNetwork(self, word_text):
        new_word = Word(word_text, 0, 0, (255, 255, 255), self)
        self._assign_seed_pos(new_word)

        for existing_word in self.words:
            weight_to_existing = self.getGematriaDistance(new_word.gematria, existing_word.gematria)
            weight_to_new = self.getGematriaDistance(existing_word.gematria, new_word.gematria)
            new_word.connect(existing_word, weight_to_existing)
            existing_word.connect(new_word, weight_to_new)

        self.words.append(new_word)
        self.referenceWords.append(new_word)
        self._markReferenceIndexDirty()
        self.makeWordCircle(self.words)
        return new_word

    def OnAddWords(self, event):
        raw_input = self.add_words_ctrl.GetValue().strip().lower()
        if raw_input == "":
            self.add_words_status.SetLabel("Type one or more words to add")
            return

        valid_chars = gematria_table.keys()
        added_words = []
        rejected_words = []

        for raw_word in raw_input.split():
            if all(char in valid_chars for char in raw_word):
                added_words.append(self.addWordToNetwork(raw_word))
            else:
                rejected_words.append(raw_word)

        self.add_words_ctrl.SetValue("")

        if added_words:
            added_info = ", ".join(f"{word.word}:{word.gematria}" for word in added_words)
            if rejected_words:
                self.add_words_status.SetLabel(f"Added {added_info}. Rejected: {' '.join(rejected_words)}")
            else:
                self.add_words_status.SetLabel(f"Added {added_info}")
        else:
            self.add_words_status.SetLabel(f"Rejected (invalid chars): {' '.join(rejected_words)}")

    def OnClearSentence(self, event):
        removed_count = len(self.words)
        removed_words = {word.word for word in self.words}
        self.words = []
        if removed_words:
            self.referenceWords = [word for word in self.referenceWords if word.word not in removed_words]
            self._markReferenceIndexDirty()
        self.last_audio_update = 0
        sanasyna.stop()
        self.audio_playing = False
        self.add_words_status.SetLabel(f"Cleared sentence ({removed_count} words removed, refs updated)")
    
    def getGematriaDistance(self, gematria1, gematria2):
        return abs(gematria1 - gematria2) / 1000
    
    def findWord(self, word, referenceWords):
        if not referenceWords:
            return None

        use_pos = self._is_pos_matching_enabled()
        source_gematria = word.gematria
        source_reduction = numerological_reduction(source_gematria)
        source_root = digital_root(source_gematria)

        self._ensureReferenceIndex(include_pos=use_pos)
        index = self.reference_index
        fluid_gematria_enabled = self._is_fluid_gematria_enabled()

        candidate_map = {}

        def _add_candidates(candidates):
            for candidate in candidates:
                candidate_map[id(candidate)] = candidate

        if use_pos:
            if self._is_fluid_pos_enabled():
                source_pos = self.getWordPOS(word.word, force=True)
            else:
                source_pos = self._get_seed_pos(word)
            _add_candidates(index["pos_gematria"].get((source_pos, source_gematria), []))
            if fluid_gematria_enabled:
                _add_candidates(index["pos_reduction"].get((source_pos, source_reduction), []))
                _add_candidates(index["pos_root"].get((source_pos, source_root), []))

        if use_pos and self._is_fluid_pos_enabled():
            has_non_self_candidate = any(candidate.word != word.word for candidate in candidate_map.values())
            if not has_non_self_candidate:
                _add_candidates(index["gematria"].get(source_gematria, []))
                if fluid_gematria_enabled:
                    _add_candidates(index["reduction"].get(source_reduction, []))
                    _add_candidates(index["root"].get(source_root, []))

        if not candidate_map:
            _add_candidates(index["gematria"].get(source_gematria, []))
            if fluid_gematria_enabled:
                _add_candidates(index["reduction"].get(source_reduction, []))
                _add_candidates(index["root"].get(source_root, []))

        jump_probability = min(1.0, max(0.0, float(self.params.get("jump_probability", 0.08))))
        jump_radius = max(0, int(self.params.get("jump_radius", 120)))
        should_jump = random.random() < jump_probability
        fluid_root_enabled = self._is_fluid_root_enabled()

        if should_jump and fluid_gematria_enabled:
            gematria_keys = list(index["gematria"].keys())
            if gematria_keys:
                nearest_keys = sorted(gematria_keys, key=lambda key: abs(key - source_gematria))
                max_jump_buckets = max(6, int(self.params.get("selection_top_k", 4)) * 3)

                selected_keys = [key for key in nearest_keys if abs(key - source_gematria) <= jump_radius]
                if not fluid_root_enabled:
                    selected_keys = [key for key in selected_keys if digital_root(key) == source_root]

                selected_keys = selected_keys[:max_jump_buckets]
                if not selected_keys:
                    fallback_keys = nearest_keys
                    if not fluid_root_enabled:
                        fallback_keys = [key for key in fallback_keys if digital_root(key) == source_root]
                    selected_keys = fallback_keys[:max_jump_buckets]

                for key in selected_keys:
                    _add_candidates(index["gematria"].get(key, []))

        candidates = list(candidate_map.values())

        if not fluid_gematria_enabled:
            should_jump = False
            candidates = [candidate for candidate in candidates if candidate.gematria == source_gematria]

        if should_jump and not fluid_root_enabled:
            has_same_root_alternative = any(
                candidate.word != word.word and digital_root(candidate.gematria) == source_root
                for candidate in candidates
            )
            if not has_same_root_alternative:
                should_jump = False

        return self._selectBestReference(word, candidates, force_jump=should_jump)
        
    def changeWord(self, word, referenceWords):
        refWord = self.findWord(word, referenceWords)
        if refWord == None:
            return False
        else:
            old_word = word.word
            old_gematria = word.gematria
            word.word = refWord.word
            word.gematria = refWord.gematria
            word.neuron.word = refWord.word
            word.neuron.gematria = refWord.gematria
            self._markReferenceIndexDirty()
            for connection in word.neuron.connections:
                if self.params["set_weight_by_gematria"] == True:
                    target_neuron = connection[0]
                    target_gematria = getattr(target_neuron, "gematria", None)
                    if target_gematria is None:
                        target_word = getattr(target_neuron, "word", "")
                        if isinstance(target_word, str) and target_word != "":
                            target_gematria = get_gematria(target_word)
                            target_neuron.gematria = target_gematria
                        else:
                            target_gematria = word.gematria
                    connection[1] = self.getGematriaDistance(word.gematria, target_gematria)
            return old_word != word.word or old_gematria != word.gematria

    def iterateSentenceToLogic(self, max_iterations):
        if not self.words or not self.referenceWords:
            return False

        threshold = self.params["word_change_threshold"]
        changed_any = False
        seen_states = set()

        for _ in range(max_iterations):
            sentence_state = tuple(word.word for word in self.words)
            if sentence_state in seen_states:
                break
            seen_states.add(sentence_state)

            changed_this_round = False
            for word in self.words:
                if word.neuron.activation < -threshold or word.neuron.activation > threshold:
                    if self.changeWord(word, self.referenceWords):
                        changed_this_round = True

            if not changed_this_round:
                break
            changed_any = True

        return changed_any

    def _run_logic_iteration_task(self, max_iterations):
        changed_any = False
        try:
            if self.running and not self.closed:
                changed_any = self.iterateSentenceToLogic(max_iterations)
        except Exception:
            changed_any = False
        finally:
            with self._logic_state_lock:
                self._logic_inflight = False
                if changed_any:
                    self._logic_changed_ready = True

    def _update_logic_worker_status_label(self):
        with self._logic_state_lock:
            busy = bool(self._logic_inflight)
            changed_pending = bool(self._logic_changed_ready)
        if busy:
            status = "Logic worker: busy"
        elif changed_pending:
            status = "Logic worker: changed ready"
        else:
            status = "Logic worker: idle"
        if getattr(self, "logic_worker_status_label", None) is not None:
            self.logic_worker_status_label.SetLabel(status)
        
    def writeToFile(self, word):
        self.outfile.write(word + " ")

    def simulationStep(self):
        now = time.time()
        melody_from_own_time = bool(self.params.get("melody_from_own_time", True))
        if now - self.last_process_time < self.params["process_interval"]:
            return
        self.last_process_time = now
        self._update_logic_worker_status_label()

        if not melody_from_own_time:
            # Tight timed-mode behavior: end the current loop exactly at period boundary
            # and force a fresh synthesis period regardless of previous signature.
            sanasyna.stop()
            self.audio_playing = False
            self.last_audio_sentence_signature = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                self.OnClose(None)
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F11:
                    self._apply_fullscreen_mode(not self._is_effective_fullscreen())
                elif event.key == pygame.K_ESCAPE and self._is_effective_fullscreen():
                    self._apply_fullscreen_mode(False)

        self._sync_display_window_state()

        self.screen.fill((0, 0, 0))
        logic_triggered = False

        for word in self.words:
            word.activate(math.sin(time.time()))
            total_activation = 0
            for connection in word.neuron.connections:
                total_activation += connection[0].activation * connection[1]
            connection_count = len(word.neuron.connections)
            if connection_count > 0:
                total_activation /= connection_count
            word.neuron.backpropagate(target=float(self.params.get("target", 0.0)))

            if word.neuron.activation < -2 or word.neuron.activation > 2:
                word.neuron.activation = 1

            if word.neuron.activation < -self.params["word_change_threshold"] or word.neuron.activation > self.params["word_change_threshold"]:
                logic_triggered = True

        sentChanged = False
        with self._logic_state_lock:
            if self._logic_changed_ready:
                sentChanged = True
                self._logic_changed_ready = False

        if logic_triggered:
            start_logic_worker = False
            with self._logic_state_lock:
                if not self._logic_inflight:
                    self._logic_inflight = True
                    start_logic_worker = True
            if start_logic_worker:
                max_iterations = self.params.get("logic_iteration_limit", 48)
                self._logic_thread = threading.Thread(
                    target=self._run_logic_iteration_task,
                    args=(max_iterations,),
                    daemon=True,
                )
                self._logic_thread.start()

        sentence = " ".join(word.word for word in self.words)

        self.updateAudio()

        #draw connections
        for word in self.words:
            for connection in word.neuron.connections:
                self.conn_color_r = int(255 * abs(connection[1]))
                self.conn_color_g = connection[0].activation * 255
                self.conn_color_b = -connection[0].activation * 255

                if self.conn_color_r < 0:
                    self.conn_color_r = 0
                if self.conn_color_g < 0:
                    self.conn_color_g = 0
                if self.conn_color_b < 0:
                    self.conn_color_b = 0
                if self.conn_color_r > 255:
                    self.conn_color_r = 255
                if self.conn_color_g > 255:
                    self.conn_color_g = 255
                if self.conn_color_b > 255:
                    self.conn_color_b = 255

                pygame.draw.line(self.screen, (self.conn_color_r, self.conn_color_g, self.conn_color_b), (word.x, word.y), (connection[0].x, connection[0].y), 5)

        for word in self.words:
            word.draw(self.screen)

        if (sentChanged):
            self.writeToFile(sentence+"\n")
            sentence_gematria = 0
            word_gematria = 0
            gematria_terms = []
            for word in sentence.split():
                word_gematria = get_gematria(word)
                sentence_gematria += word_gematria
                gematria_terms.append(f"{word}:{word_gematria}")
                self.writeToFile(str(word_gematria) + " + ")
            self.writeToFile(" = " + str(sentence_gematria))
            self.writeToFile(" -> ")
            nr_reduction_array = []
            starting_gematria = sentence_gematria
            while sentence_gematria >= 10:
                sentence_gematria = numerological_reduction(sentence_gematria)
                nr_reduction_array.append(sentence_gematria)

            for i in range(len(nr_reduction_array)):
                self.writeToFile(str(nr_reduction_array[i]))
                if i < len(nr_reduction_array) - 1:
                    self.writeToFile(" -> ")
            self.writeToFile("\n")
            self.outfile.flush()      

            self.last_result_sentence = sentence.strip()
            self.last_result_gematria_line = " + ".join(gematria_terms) + f" = {starting_gematria}"
            if nr_reduction_array:
                reduction_chain = " -> ".join(str(value) for value in nr_reduction_array)
                self.last_result_reduction_line = f"Reduction: {starting_gematria} -> {reduction_chain}"
            else:
                self.last_result_reduction_line = f"Reduction: {starting_gematria}"

            if bool(self.params.get("piper_tts_on", False)):
                self._speak_sentence_with_piper_async(self.last_result_sentence)

        if self.last_result_sentence != "":
            draw_text_centered(self.screen, self.last_result_sentence, 18, (0, 255, 127), self.size[0]/2, 20)
        if self.last_result_gematria_line != "":
            draw_text_centered(self.screen, self.last_result_gematria_line, 14, (180, 230, 255), self.size[0]/2, 40)
        if self.last_result_reduction_line != "":
            draw_text_centered(self.screen, self.last_result_reduction_line, 12, (220, 200, 255), self.size[0]/2, 56)

        pygame.display.flip()

    def testNeurons(self):
        while self.running:
            self.simulationStep()
            self.clock.tick(60)

    def OnTimer(self, event):
        if not self.running:
            self.OnClose(None)
            return
        self.simulationStep()

    def runMacMainLoop(self):
        self.timer = wx.Timer(self.frame)
        self.frame.Bind(wx.EVT_TIMER, self.OnTimer, self.timer)
        self.timer.Start(16)
        self.app.MainLoop()
        

def get_gematria(word):
    gematria = 0
    for letter in word:
        gematria += gematria_table[letter]
    return gematria

def numerological_reduction(num):
    num = sum(int(digit) for digit in str(num))
    return num

def digital_root(num):
    while num >= 10:
        num = numerological_reduction(num)
    return num

def get_distance(x1, y1, x2, y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

def get_angle(x1, y1, x2, y2):
    return math.atan2(y2-y1, x2-x1)

def get_activation_color(activation):
    if activation > 0:
        return (255, 0, 0)
    elif activation < 0:
        return (0, 0, 255)
    else:
        return (255, 255, 255)


BITMAP_FONT_5x7 = {
    "a": ("00000", "01110", "00001", "01111", "10001", "10011", "01101"),
    "b": ("10000", "10000", "10110", "11001", "10001", "11001", "10110"),
    "c": ("00000", "01110", "10001", "10000", "10000", "10001", "01110"),
    "d": ("00001", "00001", "01101", "10011", "10001", "10011", "01101"),
    "e": ("00000", "01110", "10001", "11111", "10000", "10001", "01110"),
    "f": ("00110", "01001", "01000", "11100", "01000", "01000", "01000"),
    "g": ("00000", "01101", "10011", "10011", "01101", "00001", "01110"),
    "h": ("10000", "10000", "10110", "11001", "10001", "10001", "10001"),
    "i": ("00100", "00000", "01100", "00100", "00100", "00100", "01110"),
    "j": ("00010", "00000", "00110", "00010", "00010", "10010", "01100"),
    "k": ("10000", "10000", "10011", "10100", "11100", "10010", "10001"),
    "l": ("01100", "00100", "00100", "00100", "00100", "00100", "01110"),
    "m": ("00000", "11010", "10101", "10101", "10101", "10101", "10101"),
    "n": ("00000", "10110", "11001", "10001", "10001", "10001", "10001"),
    "o": ("00000", "01110", "10001", "10001", "10001", "10001", "01110"),
    "p": ("00000", "10110", "11001", "11001", "10110", "10000", "10000"),
    "q": ("00000", "01101", "10011", "10011", "01101", "00001", "00001"),
    "r": ("00000", "10110", "11001", "10000", "10000", "10000", "10000"),
    "s": ("00000", "01111", "10000", "01110", "00001", "00001", "11110"),
    "t": ("01000", "01000", "11110", "01000", "01000", "01001", "00110"),
    "u": ("00000", "10001", "10001", "10001", "10001", "10011", "01101"),
    "v": ("00000", "10001", "10001", "10001", "10001", "01010", "00100"),
    "w": ("00000", "10001", "10001", "10101", "10101", "10101", "01010"),
    "x": ("00000", "10001", "01010", "00100", "01010", "10001", "10001"),
    "y": ("00000", "10001", "10001", "10011", "01101", "00001", "01110"),
    "z": ("00000", "11111", "00010", "00100", "01000", "10000", "11111"),
    "å": ("00100", "01010", "00100", "01110", "10001", "10001", "01110"),
    "ä": ("01010", "00000", "01110", "00001", "01111", "10001", "01111"),
    "ö": ("01010", "00000", "01110", "10001", "10001", "10001", "01110"),
    "0": ("01110", "10011", "10101", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("01110", "10000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00001", "01110"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    "+": ("00000", "00100", "00100", "11111", "00100", "00100", "00000"),
    "=": ("00000", "11111", "00000", "11111", "00000", "00000", "00000"),
    ">": ("10000", "01000", "00100", "00010", "00100", "01000", "10000"),
    "<": ("00001", "00010", "00100", "01000", "00100", "00010", "00001"),
    "/": ("00001", "00010", "00100", "01000", "10000", "00000", "00000"),
    "_": ("00000", "00000", "00000", "00000", "00000", "00000", "11111"),
    ",": ("00000", "00000", "00000", "00000", "00110", "00100", "01000"),
    ".": ("00000", "00000", "00000", "00000", "00000", "01100", "01100"),
    ":": ("00000", "01100", "01100", "00000", "01100", "01100", "00000"),
    ";": ("00000", "01100", "01100", "00000", "01100", "00100", "01000"),
    "(": ("00010", "00100", "01000", "01000", "01000", "00100", "00010"),
    ")": ("01000", "00100", "00010", "00010", "00010", "00100", "01000"),
    "[": ("01110", "01000", "01000", "01000", "01000", "01000", "01110"),
    "]": ("01110", "00010", "00010", "00010", "00010", "00010", "01110"),
    "!": ("00100", "00100", "00100", "00100", "00100", "00000", "00100"),
    "?": ("01110", "10001", "00001", "00010", "00100", "00000", "00100"),
    " ": ("00000", "00000", "00000", "00000", "00000", "00000", "00000"),
}


def draw_bitmap_text_centered(screen, text, size, color, x, y):
    text_value = str(text).lower()
    scale = max(1, int(size) // 8)
    default_glyph = BITMAP_FONT_5x7["?"]
    base_width = len(default_glyph[0])
    base_height = len(default_glyph)
    glyph_width = base_width * scale
    glyph_height = base_height * scale
    spacing = scale

    total_width = 0
    for character in text_value:
        glyph = BITMAP_FONT_5x7.get(character, default_glyph)
        char_width = len(glyph[0])
        if character == " ":
            total_width += max(2, char_width - 2) * scale + spacing
        else:
            total_width += char_width * scale + spacing
    if total_width > 0:
        total_width -= spacing

    cursor_x = int(x) - total_width // 2
    top_y = int(y) - glyph_height // 2

    for character in text_value:
        glyph = BITMAP_FONT_5x7.get(character, default_glyph)
        char_width = len(glyph[0])
        local_width = max(2, char_width - 2) * scale if character == " " else char_width * scale

        if character != " ":
            for row_index, row_pattern in enumerate(glyph):
                for col_index, pixel in enumerate(row_pattern):
                    if pixel == "1":
                        px = cursor_x + col_index * scale
                        py = top_y + row_index * scale
                        pygame.draw.rect(screen, color, (px, py, scale, scale))

        cursor_x += local_width + spacing


def draw_text_centered(screen, text, size, color, x, y):
    if pygame_font is not None:
        try:
            if not pygame_font.get_init():
                pygame_font.init()
            font_size = max(8, int(size))
            font = _pygame_mono_font_cache.get(font_size)
            if font is None:
                font = None
                for font_name in MONOSPACE_FONT_CANDIDATES:
                    font_path = pygame_font.match_font(font_name)
                    if font_path:
                        font = pygame_font.Font(font_path, font_size)
                        break
                if font is None:
                    font = pygame_font.Font(None, font_size)
                _pygame_mono_font_cache[font_size] = font
            rendered_text = font.render(str(text), 1, color)
            textpos = rendered_text.get_rect(centerx=x, centery=y)
            screen.blit(rendered_text, textpos)
            return
        except Exception:
            pass

    draw_bitmap_text_centered(screen, text, size, color, x, y)
    
class Neuron:
    def __init__(self, x, y, radius, color):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.activation = 1
        self.connections = []
        self.controller = None

    def setController(self, controller):
        self.controller = controller

    def draw(self, screen):
        self.color = get_activation_color(self.activation)
        pygame.draw.circle(screen, self.color, (self.x, self.y), self.radius)
        draw_text_centered(screen, self.activation, 16, (255, 255, 255), self.x, self.y)

    def move(self, dx, dy):
        self.x += dx
        self.y += dy

    def connect(self, neuron, weight):
        self.connections.append([neuron, weight])

    def activate(self, value):
        self.activation += sign(self.activation)*self.controller.params["activation_increase"]
        if self.activation > self.controller.params["activation_limit"]:
            self.activation = self.controller.params["activation_limit"]
        if self.activation < -self.controller.params["activation_limit"]:
            self.activation = -self.controller.params["activation_limit"]
        
        for connection in self.connections:
            self.activation += sigmoid(connection[0].activation, self.controller.params["sigmoid_scale"], self.controller) * connection[1] * value
        if self.activation >= math.inf:
            self.activation = math.inf
        if self.activation <= -math.inf:
            self.activation = -math.inf


    def backpropagate(self, target=0):
        if self.controller is None:
            return

        learning_rate = float(self.controller.params.get("learning_rate", 0.1))
        error_gain = float(self.controller.params.get("error", 0.0))
        target_value = float(target)

        prediction_error = (target_value - self.activation) * error_gain
        if prediction_error == 0.0:
            return

        sigmoid_scale = float(self.controller.params.get("sigmoid_scale", 2.0))
        weight_limit = max(0.1, float(self.controller.params.get("activation_limit", 2.0)) * 4.0)

        for connection in self.connections:
            input_neuron = connection[0]
            input_signal = sigmoid(input_neuron.activation, sigmoid_scale, self.controller)
            connection[1] += learning_rate * prediction_error * input_signal

            if connection[1] > weight_limit:
                connection[1] = weight_limit
            elif connection[1] < -weight_limit:
                connection[1] = -weight_limit

def sigmoid(x, scale, controller=None):
    scale_start = -scale
    scale_end = scale

    x *= scale
    try:
        ans = (2 / (1 + math.exp(-x)) - 1) * (scale_end - scale_start) + scale_start
    except OverflowError:
        if x > 0:
            ans = controller.params["activation_limit"]
        else:
            ans = -controller.params["activation_limit"]

    return ans


def sign(num):
    if num > 0:
        return 1
    elif num < 0:
        return -1
    else:
        return 0
    

class Word:
    def __init__(self, word, x, y, color, controller=None):
        self.controller = controller
        self.word = word
        self.gematria = get_gematria(word)
        self.seed_pos = ""
        self.x = x
        self.y = y
        self.neuron = Neuron(x, y, 20, color)
        self.neuron.word = word
        self.neuron.gematria = self.gematria
        self.neuron.setController(controller)

    def getConnectedWordsLabel(self, max_words=4):
        if not self.neuron.connections:
            return ""

        weighted_names = []
        for connection in self.neuron.connections:
            connected_neuron = connection[0]
            weight = connection[1]
            if hasattr(connected_neuron, "word"):
                weighted_names.append((connected_neuron.word, abs(weight)))

        if not weighted_names:
            return ""

        weighted_names.sort(key=lambda item: item[1], reverse=True)
        ordered_names = []
        for name, _ in weighted_names:
            if name not in ordered_names:
                ordered_names.append(name)

        visible_names = ordered_names[:max_words]
        hidden_count = max(0, len(ordered_names) - len(visible_names))
        label = "-> " + ", ".join(visible_names)
        if hidden_count > 0:
            label += f" (+{hidden_count})"
        return label

    def getPOSLabel(self):
        if self.controller is None:
            return ""
        if not self.controller._is_pos_matching_enabled():
            return ""
        try:
            if self.controller._is_fluid_pos_enabled():
                return self.controller.getWordPOS(self.word, force=True)
            return self.controller._get_seed_pos(self)
        except Exception:
            return ""

    def draw(self, screen):
        self.neuron.draw(screen)
        draw_text_centered(screen, self.word, 22, get_activation_color(self.neuron.activation), self.x, self.y + 30)
        pos_label = self.getPOSLabel()
        if pos_label != "":
            draw_text_centered(screen, f"POS:{pos_label}", 12, (220, 220, 140), self.x, self.y + 46)
        connected_words_label = self.getConnectedWordsLabel()
        if connected_words_label != "":
            draw_text_centered(screen, connected_words_label, 12, (170, 170, 170), self.x, self.y + 60)

    def move(self, dx, dy):
        self.x += dx
        self.y += dy
        self.neuron.move(dx, dy)
        if self.x < 100:
            self.x = 100
        if self.x > 700:
            self.x = 700
        if self.y < 100:
            self.y = 100
        if self.y > 500:
            self.y = 500

    def connect(self, word, weight):
        if self != word:
            self.neuron.connect(word.neuron, weight)

    def activate(self, value):
        self.neuron.activate(value)


if __name__ == "__main__":
    kontrol = SanaVerkkoKontrolleri()
    if sys.platform == "darwin":
        kontrol.runMacMainLoop()
    else:
        kontrol_thread = threading.Thread(target=kontrol.testNeurons)
        kontrol_thread.start()
        kontrol.app.MainLoop()
        kontrol_thread.join()

