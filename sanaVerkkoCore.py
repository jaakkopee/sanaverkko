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
        self.learning_rate_ctrl = wx.TextCtrl(panel, -1, str(self.params["learning_rate"]))
        self.learning_rate_ctrl.Bind(wx.EVT_TEXT, self.OnLearningRate)

        self.error_label = wx.StaticText(panel, -1, "Error")
        self.error_ctrl = wx.TextCtrl(panel, -1, str(self.params["error"]))
        self.error_ctrl.Bind(wx.EVT_TEXT, self.OnError)

        self.activation_increase_label = wx.StaticText(panel, -1, "Activation increase")
        self.activation_increase_ctrl = wx.TextCtrl(panel, -1, str(self.params["activation_increase"]))
        self.activation_increase_ctrl.Bind(wx.EVT_TEXT, self.OnActivationIncrease)

        self.activation_limit_label = wx.StaticText(panel, -1, "Activation limit")
        self.activation_limit_ctrl = wx.TextCtrl(panel, -1, str(self.params["activation_limit"]))
        self.activation_limit_ctrl.Bind(wx.EVT_TEXT, self.OnActivationLimit)

        self.sigmoid_scale_label = wx.StaticText(panel, -1, "Sigmoid scale")
        self.sigmoid_scale_ctrl = wx.TextCtrl(panel, -1, str(self.params["sigmoid_scale"]))
        self.sigmoid_scale_ctrl.Bind(wx.EVT_TEXT, self.OnSigmoidScale)

        self.word_change_threshold_label = wx.StaticText(panel, -1, "Word change threshold")
        self.word_change_threshold_ctrl = wx.TextCtrl(panel, -1, str(self.params["word_change_threshold"]))
        self.word_change_threshold_ctrl.Bind(wx.EVT_TEXT, self.OnWordChangeThreshold)

        self.zoom_label = wx.StaticText(panel, -1, "Zoom")
        self.zoom_ctrl = wx.TextCtrl(panel, -1, str(self.params["zoom"]))
        self.zoom_ctrl.Bind(wx.EVT_TEXT, self.OnZoom)

        self.add_words_label = wx.StaticText(panel, -1, "Add word(s)")
        self.add_words_ctrl = wx.TextCtrl(panel, -1, "", style=wx.TE_PROCESS_ENTER)
        self.add_words_ctrl.Bind(wx.EVT_TEXT_ENTER, self.OnAddWords)
        self.add_words_button = wx.Button(panel, -1, "Add")
        self.add_words_button.Bind(wx.EVT_BUTTON, self.OnAddWords)
        self.add_words_status = wx.StaticText(panel, -1, "")

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

        add_words_row = wx.BoxSizer(wx.HORIZONTAL)
        add_words_row.Add(self.add_words_ctrl, 1, wx.RIGHT, 5)
        add_words_row.Add(self.add_words_button, 0)
        self.sizer.Add(self.add_words_label, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.sizer.Add(add_words_row, 0, wx.EXPAND | wx.ALL, 5)
        self.sizer.Add(self.add_words_status, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)

        panel.SetSizer(self.sizer)
        self.app.SetTopWindow(self.frame)
        self.frame.Show()

    def OnSetWeightByGematria(self, event):
        self.params["set_weight_by_gematria"] = self.set_weight_by_gematria_checkbox.GetValue()

    def OnLearningRate(self, event):
        self.params["learning_rate"] = float(self.learning_rate_ctrl.GetValue())

    def OnError(self, event):
        self.params["error"] = float(self.error_ctrl.GetValue())

    def OnActivationIncrease(self, event):
        self.params["activation_increase"] = float(self.activation_increase_ctrl.GetValue())

    def OnActivationLimit(self, event):
        self.params["activation_limit"] = float(self.activation_limit_ctrl.GetValue())

    def OnSigmoidScale(self, event):
        self.params["sigmoid_scale"] = float(self.sigmoid_scale_ctrl.GetValue())

    def OnWordChangeThreshold(self, event):
        self.params["word_change_threshold"] = float(self.word_change_threshold_ctrl.GetValue())

    def OnZoom(self, event):
        self.params["zoom"] = float(self.zoom_ctrl.GetValue())
        self.makeWordCircle(self.words)

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
        sanasyna.init_audio(self.audio_sample_rate)

    def updateAudio(self):
        now = time.time()
        if now - self.last_audio_update < self.audio_refresh_interval:
            return
        self.last_audio_update = now

        if len(self.words) == 0:
            return

        average_activation = sum(abs(word.neuron.activation) for word in self.words) / len(self.words)
        gematria_total = sum(word.gematria for word in self.words)
        frequency = 110 + (gematria_total % 770)
        amplitude = min(0.25, max(0.05, average_activation / 10))

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
        input_filename = sys.argv[1] if len(sys.argv) > 1 else "input.txt"
        reference_filename = sys.argv[2] if len(sys.argv) > 2 else input_filename

        self.words = self.parseText(input_filename)
        self.referenceWords = self.parseText(reference_filename)

        for word in self.words:
            self.referenceWords.append(word)

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
        file = open(filename, "r")
        words = []
        valid_chars = gematria_table.keys()
        for line in file:
            for word in line.split():
                word = word.lower()
                if all(char in valid_chars for char in word):
                    words.append(Word(word, 0, 0, (255, 255, 255), self))
        return words

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
            for connection in word.neuron.connections:
                connection[0].word = refWord.word
                connection[0].gematria = refWord.gematria
                if self.params["set_weight_by_gematria"] == True:
                    connection[1] = self.getGematriaDistance(word.gematria, connection[0].gematria)
                
            return word
        
    def writeToFile(self, word):
        self.outfile.write(word + " ")

    def simulationStep(self):
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
            total_activation /= len(word.neuron.connections)
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
            for word in sentence.split():
                word_gematria = get_gematria(word)
                sentence_gematria += word_gematria
                self.writeToFile(str(word_gematria) + " + ")
            self.writeToFile(" = " + str(sentence_gematria))
            self.writeToFile(" -> ")
            nr_reduction_array = []
            while sentence_gematria >= 10:
                sentence_gematria = numerological_reduction(sentence_gematria)
                nr_reduction_array.append(sentence_gematria)

            for i in range(len(nr_reduction_array)):
                self.writeToFile(str(nr_reduction_array[i]))
                if i < len(nr_reduction_array) - 1:
                    self.writeToFile(" -> ")
            self.writeToFile("\n")
            self.outfile.flush()      

            #draw sentence
            draw_text_centered(self.screen, sentence, 18, (0, 255, 127), self.size[0]/2, 20)

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


def draw_text_centered(screen, text, size, color, x, y):
    if pygame_font is None:
        return

    try:
        if not pygame_font.get_init():
            pygame_font.init()
        font = pygame_font.Font(None, size)
        rendered_text = font.render(str(text), 1, color)
        textpos = rendered_text.get_rect(centerx=x, centery=y)
        screen.blit(rendered_text, textpos)
    except Exception:
        return
    
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
        self.neuron.setController(controller)

    def draw(self, screen):
        self.neuron.draw(screen)
        draw_text_centered(screen, self.word, 22, get_activation_color(self.neuron.activation), self.x, self.y + 30)

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

