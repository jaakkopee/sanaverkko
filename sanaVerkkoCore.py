import numpy as np
import pygame
import random
import time
import math
import wx
import threading
import sys
import sanasyna

try:
    import pygame.font as pygame_font
except Exception:
    pygame_font = None



gematria_table = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7, "h": 8, "i": 9, "j": 10, "k": 20, "l": 30, "m": 40, "n": 50, "o": 60, "p": 70, "q": 80, "r": 90, "s": 100, "t": 200, "u": 300, "v": 400, "w": 500, "x": 600, "y": 700, "z": 800, "å": 900, "ä": 1000, "ö": 1100}

class SanaVerkkoKontrolleri:
    
    def __init__(self):
        self.params = {}
        self.params["set_weight_by_gematria"] = False
        self.params["learning_rate"] = 0.1
        self.params["error"] = 0
        self.params["target"] = 0
        self.params["activation_increase"] = 0.0001
        self.params["activation_limit"] = 2
        self.params["sigmoid_scale"] = 2
        self.params["word_change_threshold"] = 0.777
        self.params["zoom"]=0.1
        self.params["process_interval"] = 0.25
        self.params["import_mode"] = "append"
        self.params["audio_wave_mode"] = "dynamic"

        self.app = wx.App()

        self.frame = wx.Frame(None, -1, "SanaVerkko")
        self.frame.Bind(wx.EVT_CLOSE, self.OnClose)
        self.frame.SetSize(600, 800)
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

        self.conn_color_r = 0
        self.conn_color_g = 0
        self.conn_color_b = 0
        self.last_process_time = 0
        self.last_result_sentence = ""
        self.last_result_gematria_line = ""
        self.last_result_reduction_line = ""

        self.initPygame()
        self.initAudio()
        self.initWords()
        self.widgetSetup()
        self.outfile = open("output.txt", "w")

    def widgetSetup(self):
        panel = wx.Panel(self.frame, -1)
        self.set_weight_by_gematria_checkbox = wx.CheckBox(panel, -1, "Set weight by gematria")
        self.set_weight_by_gematria_checkbox.SetValue(self.params["set_weight_by_gematria"])
        self.set_weight_by_gematria_checkbox.Bind(wx.EVT_CHECKBOX, self.OnSetWeightByGematria)

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

        self.audio_wave_mode_label = wx.StaticText(panel, -1, "Audio waveform mode")
        self.audio_wave_mode_choice = wx.Choice(panel, -1, choices=["Dynamic", "Pure sine", "Noise-heavy", "Classic analog"])
        self.audio_wave_mode_choice.SetSelection(0)
        self.audio_wave_mode_choice.Bind(wx.EVT_CHOICE, self.OnAudioWaveMode)

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

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.set_weight_by_gematria_checkbox, 0, wx.ALL, 5)

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
        self.sizer.Add(self.process_interval_ctrl, 0, wx.ALL, 5)

        self.sizer.Add(self.audio_wave_mode_label, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.sizer.Add(self.audio_wave_mode_choice, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)

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

        panel.SetSizer(self.sizer)
        self.app.SetTopWindow(self.frame)
        self.frame.Show()

    def OnSetWeightByGematria(self, event):
        self.params["set_weight_by_gematria"] = self.set_weight_by_gematria_checkbox.GetValue()

    def _bindNumericCtrl(self, ctrl, handler):
        ctrl.Bind(wx.EVT_TEXT_ENTER, handler)
        ctrl.Bind(wx.EVT_KILL_FOCUS, handler)

    def _readFloat(self, ctrl):
        text_value = ctrl.GetValue().strip().replace(",", ".")
        if text_value == "":
            return None
        try:
            return float(text_value)
        except ValueError:
            return None

    def OnLearningRate(self, event):
        value = self._readFloat(self.learning_rate_ctrl)
        if value is not None:
            self.params["learning_rate"] = value
        event.Skip()

    def OnError(self, event):
        value = self._readFloat(self.error_ctrl)
        if value is not None:
            self.params["error"] = value
        event.Skip()

    def OnActivationIncrease(self, event):
        value = self._readFloat(self.activation_increase_ctrl)
        if value is not None:
            self.params["activation_increase"] = value
        event.Skip()

    def OnActivationLimit(self, event):
        value = self._readFloat(self.activation_limit_ctrl)
        if value is not None:
            self.params["activation_limit"] = value
        event.Skip()

    def OnSigmoidScale(self, event):
        value = self._readFloat(self.sigmoid_scale_ctrl)
        if value is not None:
            self.params["sigmoid_scale"] = value
        event.Skip()

    def OnWordChangeThreshold(self, event):
        value = self._readFloat(self.word_change_threshold_ctrl)
        if value is not None:
            self.params["word_change_threshold"] = value
        event.Skip()

    def OnZoom(self, event):
        value = self._readFloat(self.zoom_ctrl)
        if value is not None:
            self.params["zoom"] = value
            self.makeWordCircle(self.words)
        event.Skip()

    def OnProcessInterval(self, event):
        value = self._readFloat(self.process_interval_ctrl)
        if value is not None:
            self.params["process_interval"] = max(0.01, value)
            self.process_interval_ctrl.SetValue(str(self.params["process_interval"]))
        event.Skip()

    def OnImportMode(self, event):
        selected_mode = self.import_mode_choice.GetStringSelection()
        if selected_mode == "Replace database":
            self.params["import_mode"] = "replace"
        else:
            self.params["import_mode"] = "append"
        event.Skip()

    def OnAudioWaveMode(self, event):
        selected_mode = self.audio_wave_mode_choice.GetStringSelection()
        if selected_mode == "Pure sine":
            self.params["audio_wave_mode"] = "pure_sine"
        elif selected_mode == "Noise-heavy":
            self.params["audio_wave_mode"] = "noise_heavy"
        elif selected_mode == "Classic analog":
            self.params["audio_wave_mode"] = "classic_analog"
        else:
            self.params["audio_wave_mode"] = "dynamic"
        event.Skip()

    def OnClose(self, event):
        if self.closed:
            return

        self.running = False
        self.closed = True
        if self.timer is not None:
            self.timer.Stop()
            self.timer = None
        sanasyna.stop()
        sanasyna.close()
        pygame.quit()
        if hasattr(self, "outfile") and not self.outfile.closed:
            self.outfile.close()
        if self.frame is not None:
            self.frame.Destroy()
        if self.app is not None and self.app.IsMainLoopRunning():
            self.app.ExitMainLoop()

    def getParam(self, param):
        return self.params[param]

    def setParam(self, param, value):
        self.params[param] = value

    def initPygame(self):
        pygame.init()
        self.size = width, height = 1024, 768
        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption("SanaVerkko")
        self.clock = pygame.time.Clock()

    def initAudio(self):
        self.audio_sample_rate = 22050
        self.audio_refresh_interval = 0.15
        self.last_audio_update = 0
        self.audio_playing = False
        self.audio_wave_index = 0
        self.audio_waveforms = ["sine", "triangle", "square", "sawtooth", "noise"]
        sanasyna.init_audio(self.audio_sample_rate)

    def updateAudio(self):
        now = time.time()
        if now - self.last_audio_update < self.audio_refresh_interval:
            return
        self.last_audio_update = now

        if len(self.words) == 0:
            return

        average_activation = sum(abs(word.neuron.activation) for word in self.words) / len(self.words)
        signed_activation = sum(word.neuron.activation for word in self.words) / len(self.words)
        gematria_total = sum(word.gematria for word in self.words)
        frequency = 110 + (gematria_total % 770)
        amplitude = min(0.25, max(0.05, average_activation / 10))

        activation_spread = sum(abs(word.neuron.activation - signed_activation) for word in self.words) / len(self.words)
        mode = self.params.get("audio_wave_mode", "dynamic")

        if mode == "pure_sine":
            waveform = "sine"
        elif mode == "noise_heavy":
            noise_heavy_waves = ["noise", "square", "noise", "triangle", "noise", "sawtooth"]
            self.audio_wave_index = (self.audio_wave_index + 1) % len(noise_heavy_waves)
            waveform = noise_heavy_waves[self.audio_wave_index]
        elif mode == "classic_analog":
            classic_waves = ["triangle", "sawtooth", "triangle", "square", "sawtooth"]
            self.audio_wave_index = (self.audio_wave_index + 1) % len(classic_waves)
            waveform = classic_waves[self.audio_wave_index]
        else:
            dynamic_offset = int((abs(signed_activation) + activation_spread) * 10)
            self.audio_wave_index = (self.audio_wave_index + 1 + dynamic_offset) % len(self.audio_waveforms)
            waveform = self.audio_waveforms[self.audio_wave_index]

        if waveform == "triangle":
            sanasyna.generate_triangle_wave(frequency, amplitude * 0.95, self.audio_sample_rate)
        elif waveform == "square":
            sanasyna.generate_square_wave(frequency, amplitude * 0.80, self.audio_sample_rate)
        elif waveform == "sawtooth":
            sanasyna.generate_sawtooth_wave(frequency, amplitude * 0.85, self.audio_sample_rate)
        elif waveform == "noise":
            noise_amplitude = max(0.03, amplitude * 0.55)
            sanasyna.generate_noise_wave(frequency, noise_amplitude, self.audio_sample_rate)
        else:
            sanasyna.generate_sine_wave(frequency, amplitude, self.audio_sample_rate)

        sanasyna.play(loop=True)
        self.audio_playing = True

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
            self.referenceWords.append(word)

        self.referenceWords = self._uniqueWordObjects(self.referenceWords)

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

    def importReferenceDatabase(self, filename, mode="append"):
        imported_words = self.parseText(filename)
        if not imported_words:
            return 0, len(self.referenceWords)

        if mode == "replace":
            self.referenceWords = self._uniqueWordObjects(imported_words)
        else:
            self.referenceWords = self._uniqueWordObjects(self.referenceWords + imported_words)

        self.referenceWords = self._uniqueWordObjects(self.referenceWords + self.words)
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

    def addWordToNetwork(self, word_text):
        new_word = Word(word_text, 0, 0, (255, 255, 255), self)

        for existing_word in self.words:
            weight_to_existing = self.getGematriaDistance(new_word.gematria, existing_word.gematria)
            weight_to_new = self.getGematriaDistance(existing_word.gematria, new_word.gematria)
            new_word.connect(existing_word, weight_to_existing)
            existing_word.connect(new_word, weight_to_new)

        self.words.append(new_word)
        self.referenceWords.append(new_word)
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
        self.last_audio_update = 0
        sanasyna.stop()
        self.audio_playing = False
        self.add_words_status.SetLabel(f"Cleared sentence ({removed_count} words removed, refs updated)")
    
    def getGematriaDistance(self, gematria1, gematria2):
        return abs(gematria1 - gematria2) / 1000
    
    def findWord(self, word, referenceWords):
        retwords = []

        for refWord in referenceWords:
            if word.gematria == refWord.gematria:
                retwords += [refWord]
            if numerological_reduction(word.gematria) == numerological_reduction(refWord.gematria):
                retwords += [refWord]
            if digital_root(word.gematria) == digital_root(refWord.gematria):
                retwords += [refWord]
            
        if retwords != []:
            return random.choice(retwords)
        else:
            return None
        
    def changeWord(self, word, referenceWords):
        refWord = self.findWord(word, referenceWords)
        if refWord == None:
            return word
        else:
            word.word = refWord.word
            word.gematria = refWord.gematria
            word.neuron.word = refWord.word
            for connection in word.neuron.connections:
                if self.params["set_weight_by_gematria"] == True:
                    connection[1] = self.getGematriaDistance(word.gematria, connection[0].gematria)
                
            return word
        
    def writeToFile(self, word):
        self.outfile.write(word + " ")

    def simulationStep(self):
        now = time.time()
        if now - self.last_process_time < self.params["process_interval"]:
            return
        self.last_process_time = now

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                self.OnClose(None)
                return

        self.screen.fill((0, 0, 0))
        gematria = 0
        sentence = ""
        sentChanged = False

        for word in self.words:
            word.activate(math.sin(time.time()))
            total_activation = 0
            for connection in word.neuron.connections:
                total_activation += connection[0].activation * connection[1]
            connection_count = len(word.neuron.connections)
            if connection_count > 0:
                total_activation /= connection_count
            word.neuron.backpropagate(target=0)

            if word.neuron.activation < -2 or word.neuron.activation > 2:
                word.neuron.activation = 1

            if word.neuron.activation < -self.params["word_change_threshold"] or word.neuron.activation > self.params["word_change_threshold"]:
                self.changeWord(word, self.referenceWords)
                sentChanged = True

            sentence += word.word + " "
            gematria += word.gematria

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


BITMAP_FONT_3x5 = {
    "a": ("111", "101", "111", "101", "101"),
    "b": ("110", "101", "110", "101", "110"),
    "c": ("111", "100", "100", "100", "111"),
    "d": ("110", "101", "101", "101", "110"),
    "e": ("111", "100", "110", "100", "111"),
    "f": ("111", "100", "110", "100", "100"),
    "g": ("111", "100", "101", "101", "111"),
    "h": ("101", "101", "111", "101", "101"),
    "i": ("111", "010", "010", "010", "111"),
    "j": ("111", "001", "001", "101", "111"),
    "k": ("101", "101", "110", "101", "101"),
    "l": ("100", "100", "100", "100", "111"),
    "m": ("101", "111", "111", "101", "101"),
    "n": ("101", "111", "111", "111", "101"),
    "o": ("111", "101", "101", "101", "111"),
    "p": ("111", "101", "111", "100", "100"),
    "q": ("111", "101", "101", "111", "001"),
    "r": ("111", "101", "110", "101", "101"),
    "s": ("111", "100", "111", "001", "111"),
    "t": ("111", "010", "010", "010", "010"),
    "u": ("101", "101", "101", "101", "111"),
    "v": ("101", "101", "101", "101", "010"),
    "w": ("101", "101", "111", "111", "101"),
    "x": ("101", "101", "010", "101", "101"),
    "y": ("101", "101", "111", "001", "111"),
    "z": ("111", "001", "010", "100", "111"),
    "å": ("111", "101", "111", "101", "101"),
    "ä": ("111", "101", "111", "101", "101"),
    "ö": ("111", "101", "101", "101", "111"),
    "0": ("111", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"),
    "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"),
    "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"),
    "7": ("111", "001", "010", "100", "100"),
    "8": ("111", "101", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "111"),
    "-": ("000", "000", "111", "000", "000"),
    "+": ("000", "010", "111", "010", "000"),
    ">": ("100", "010", "001", "010", "100"),
    ",": ("000", "000", "000", "010", "100"),
    ".": ("000", "000", "000", "000", "010"),
    ":": ("000", "010", "000", "010", "000"),
    "(": ("001", "010", "010", "010", "001"),
    ")": ("100", "010", "010", "010", "100"),
    " ": ("000", "000", "000", "000", "000"),
    "?": ("111", "001", "011", "000", "010"),
}


def draw_bitmap_text_centered(screen, text, size, color, x, y):
    text_value = str(text).lower()
    scale = max(1, int(size) // 6)
    glyph_width = 3 * scale
    glyph_height = 5 * scale
    spacing = scale

    total_width = 0
    for character in text_value:
        if character == " ":
            total_width += 2 * scale + spacing
        else:
            total_width += glyph_width + spacing
    if total_width > 0:
        total_width -= spacing

    cursor_x = int(x) - total_width // 2
    top_y = int(y) - glyph_height // 2

    for character in text_value:
        glyph = BITMAP_FONT_3x5.get(character, BITMAP_FONT_3x5["?"])
        local_width = 2 * scale if character == " " else glyph_width

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
            font = pygame_font.Font(None, size)
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
        learning_rate = self.controller.params["learning_rate"]
        error = (target - self.activation) * self.controller.params["error"]
        for connection in self.connections:
            connection[0].activation += connection[1] * error * learning_rate

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
        self.x = x
        self.y = y
        self.neuron = Neuron(x, y, 20, color)
        self.neuron.word = word
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

    def draw(self, screen):
        self.neuron.draw(screen)
        draw_text_centered(screen, self.word, 22, get_activation_color(self.neuron.activation), self.x, self.y + 30)
        connected_words_label = self.getConnectedWordsLabel()
        if connected_words_label != "":
            draw_text_centered(screen, connected_words_label, 14, (170, 170, 170), self.x, self.y + 46)

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

