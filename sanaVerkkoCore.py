import numpy as np
import pygame
import random
import time
import math
import wx
import threading



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

        self.app = wx.App(False)
        self.frame = wx.Frame(None, wx.ID_ANY, "SanaVerkko", size=(400, 800))
        self.panel = wx.Panel(self.frame)
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.frame.SetSizer(self.sizer)
        self.frame.Show()
        self.mainloop = threading.Thread(target=self.app.MainLoop)

        self.words = []
        self.referenceWords = []
        self.wordsToChange = []
        self.wordsToChangeIndex = 0

        self.screen = None
        self.size = None
        self.clock = None

        self.conn_color_r = 0
        self.conn_color_g = 0
        self.conn_color_b = 0

        self.initPygame()
        self.initWords()
        self.widgetSetup()
        self.outfile = open("output.txt", "w")

    def widgetSetup(self):
        self.set_weight_by_gematria_checkbox = wx.CheckBox(self.frame, label="Set weight by gematria")

        self.word_change_threshold_label = wx.StaticText(self.frame, label="Word change threshold")
        self.word_change_threshold_slider = wx.Slider(self.frame, value=0, minValue=0, maxValue=1000, style=wx.SL_HORIZONTAL)
        self.word_change_threshold_value = wx.StaticText(self.frame, label="0")

        self.learning_rate_label = wx.StaticText(self.frame, label="Learning rate")
        self.learning_rate_slider = wx.Slider(self.frame, value=0, minValue=0, maxValue=100, style=wx.SL_HORIZONTAL)
        self.learning_rate_value = wx.StaticText(self.frame, label="0")

        self.error_label = wx.StaticText(self.frame, label="Error")
        self.error_slider = wx.Slider(self.frame, value=0, minValue=0, maxValue=100, style=wx.SL_HORIZONTAL)
        self.error_value = wx.StaticText(self.frame, label="0")

        self.target_label = wx.StaticText(self.frame, label="Target")
        self.target_slider = wx.Slider(self.frame, value=0, minValue=-100, maxValue=100, style=wx.SL_HORIZONTAL)
        self.target_value = wx.StaticText(self.frame, label="0")

        self.activation_increase_label = wx.StaticText(self.frame, label="Activation increase")
        self.activation_increase_slider = wx.Slider(self.frame, value=-0, minValue=0, maxValue=500, style=wx.SL_HORIZONTAL)
        self.activation_increase_value = wx.StaticText(self.frame, label="0")

        self.activation_limit_label = wx.StaticText(self.frame, label="Activation limit")
        self.activation_limit_slider = wx.Slider(self.frame, value=5, minValue=0, maxValue=10, style=wx.SL_HORIZONTAL)
        self.activation_limit_value = wx.StaticText(self.frame, label="5")

        self.sigmoid_scale_label = wx.StaticText(self.frame, label="Sigmoid scale")
        self.sigmoid_scale_slider = wx.Slider(self.frame, value=5, minValue=0, maxValue=10, style=wx.SL_HORIZONTAL)
        self.sigmoid_scale_value = wx.StaticText(self.frame, label="5")

        self.sizer.Add(self.set_weight_by_gematria_checkbox, 0, wx.ALL, 5)
        
        self.sizer.Add(self.word_change_threshold_label, 0, wx.ALL, 5)
        self.sizer.Add(self.word_change_threshold_slider, 0, wx.ALL, 5)
        self.sizer.Add(self.word_change_threshold_value, 0, wx.ALL, 5)

        self.sizer.Add(self.learning_rate_label, 0, wx.ALL, 5)
        self.sizer.Add(self.learning_rate_slider, 0, wx.ALL, 5)
        self.sizer.Add(self.learning_rate_value, 0, wx.ALL, 5)

        self.sizer.Add(self.error_label, 0, wx.ALL, 5)
        self.sizer.Add(self.error_slider, 0, wx.ALL, 5)
        self.sizer.Add(self.error_value, 0, wx.ALL, 5)

        self.sizer.Add(self.target_label, 0, wx.ALL, 5)
        self.sizer.Add(self.target_slider, 0, wx.ALL, 5)
        self.sizer.Add(self.target_value, 0, wx.ALL, 5)

        self.sizer.Add(self.activation_increase_label, 0, wx.ALL, 5)
        self.sizer.Add(self.activation_increase_slider, 0, wx.ALL, 5)
        self.sizer.Add(self.activation_increase_value, 0, wx.ALL, 5)

        self.sizer.Add(self.activation_limit_label, 0, wx.ALL, 5)
        self.sizer.Add(self.activation_limit_slider, 0, wx.ALL, 5)
        self.sizer.Add(self.activation_limit_value, 0, wx.ALL, 5)

        self.sizer.Add(self.sigmoid_scale_label, 0, wx.ALL, 5)
        self.sizer.Add(self.sigmoid_scale_slider, 0, wx.ALL, 5)
        self.sizer.Add(self.sigmoid_scale_value, 0, wx.ALL, 5)

        self.set_weight_by_gematria_checkbox.Bind(wx.EVT_CHECKBOX, self.onSetWeightByGematriaChange)
        self.word_change_threshold_slider.Bind(wx.EVT_SCROLL, self.onWordChangeThresholdChange)
        self.learning_rate_slider.Bind(wx.EVT_SCROLL, self.onLearningRateChange)
        self.error_slider.Bind(wx.EVT_SCROLL, self.onErrorChange)
        self.target_slider.Bind(wx.EVT_SCROLL, self.onTargetChange)
        self.activation_increase_slider.Bind(wx.EVT_SCROLL, self.onActivationIncreaseChange)
        self.activation_limit_slider.Bind(wx.EVT_SCROLL, self.onActivationLimitChange)
        self.sigmoid_scale_slider.Bind(wx.EVT_SCROLL, self.onSigmoidScaleChange)

        self.sizer.Layout()

    def onSetWeightByGematriaChange(self, event):
        self.setParam("set_weight_by_gematria", event.IsChecked())

    def onWordChangeThresholdChange(self, event):
        self.setParam("word_change_threshold", event.GetPosition()/100)
        self.word_change_threshold_value.SetLabel(str(event.GetPosition()/100))

    def onLearningRateChange(self, event):
        self.setParam("learning_rate", event.GetPosition()/100)
        self.learning_rate_value.SetLabel(str(event.GetPosition()/100))

    def onErrorChange(self, event):
        self.setParam("error", event.GetPosition()/100)
        self.error_value.SetLabel(str(event.GetPosition()/100))

    def onTargetChange(self, event):
        self.setParam("target", event.GetPosition()/100)
        self.target_value.SetLabel(str(event.GetPosition()/100))

    def onActivationIncreaseChange(self, event):
        self.setParam("activation_increase", event.GetPosition()/500)
        self.activation_increase_value.SetLabel(str(event.GetPosition()/500))

    def onActivationLimitChange(self, event):
        self.setParam("activation_limit", event.GetPosition())
        self.activation_limit_value.SetLabel(str(event.GetPosition()))

    def onSigmoidScaleChange(self, event):
        self.setParam("sigmoid_scale", event.GetPosition())
        self.sigmoid_scale_value.SetLabel(str(event.GetPosition()))

    def getParam(self, param):
        return self.params[param]

    def setParam(self, param, value):
        self.params[param] = value

    def initPygame(self):
        pygame.init()
        self.size = width, height = 800, 600
        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption("SanaVerkko")
        self.clock = pygame.time.Clock()

    def initWords(self):
        self.words = self.parseText("input.txt")
        self.referenceWords = self.parseText("nsoe.txt")

        for word in self.words:
            self.referenceWords.append(word)

        #place words in a circle
        for i, word in enumerate(self.words):
            word.x = self.size[0]/2 + 200 * math.cos(2 * math.pi * i / len(self.words))
            word.y = self.size[1]/2 + 200 * math.sin(2 * math.pi * i / len(self.words))
            word.neuron.x = word.x
            word.neuron.y = word.y

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

    def testNeurons(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
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
                #print words again and their gematria
                word.draw(self.screen)
                

            if (sentChanged):
                self.writeToFile(sentence+"\n")
                print (sentence)
                #draw sentence
                font = pygame.font.Font(None, 18)
                text = font.render(sentence, 1, (0, 255, 127))
                textpos = text.get_rect(centerx=self.size[0]/2, centery=20)
                self.screen.blit(text, textpos)

            pygame.display.flip()
        

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
        font = pygame.font.Font(None, 16)
        text = font.render(str(self.activation), 1, (255, 255, 255))
        textpos = text.get_rect(centerx=self.x, centery=self.y)
        screen.blit(text, textpos)

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
            self.activation += sigmoid(connection[0].activation, self.controller.params["sigmoid_scale"]) * connection[1] * value
        if self.activation >= math.inf:
            self.activation = math.inf
        if self.activation <= -math.inf:
            self.activation = -math.inf


    def backpropagate(self, target=0):
        learning_rate = self.controller.params["learning_rate"]
        error = (target - self.activation) * self.controller.params["error"]
        for connection in self.connections:
            connection[0].activation += connection[1] * error * learning_rate

def sigmoid(x, scale):
    scale_start = -scale
    scale_end = scale

    x *= scale
    ans = (2 / (1 + math.exp(-x)) - 1) * (scale_end - scale_start) + scale_start
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
        font = pygame.font.Font(None, 22)
        text = font.render(self.word, 1, get_activation_color(self.neuron.activation))
        textpos = text.get_rect(centerx=self.x, centery=self.y + 30)
        screen.blit(text, textpos)

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
    kontrol.testNeurons()

