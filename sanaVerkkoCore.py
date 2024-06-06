import numpy as np
import pygame
import random
import time
import math
import tkinter as tk
import threading
import sys



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

        self.tk = tk.Tk()
        self.frame = tk.Frame(self.tk)
        self.frame.pack()
        self.sizer = tk.Grid()
        self.frame.config(width=400, height=600)
        self.frame.grid()
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(0, weight=1)
        self.frame.pack_propagate(False)
        self.frame.pack()


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
        self.set_weight_by_gematria_checkbox = tk.Checkbutton(self.frame, text="Set weight by gematria", variable=self.params["set_weight_by_gematria"])

        self.word_change_threshold_label = tk.Label(self.frame, text="Word change threshold")
        self.word_change_threshold_slider = tk.Scale(self.frame, from_=0, to=100, orient=tk.HORIZONTAL)
        self.word_change_threshold_value = tk.Label(self.frame, text="0")
        
        self.learning_rate_label = tk.Label(self.frame, text="Learning rate")
        self.learning_rate_slider = tk.Scale(self.frame, from_=0, to=100, orient=tk.HORIZONTAL)
        self.learning_rate_value = tk.Label(self.frame, text="0")

        self.error_label = tk.Label(self.frame, text="Error")
        self.error_slider = tk.Scale(self.frame, from_=0, to=100, orient=tk.HORIZONTAL)
        self.error_value = tk.Label(self.frame, text="0")

        self.target_label = tk.Label(self.frame, text="Target")
        self.target_slider = tk.Scale(self.frame, from_=0, to=100, orient=tk.HORIZONTAL)
        self.target_value = tk.Label(self.frame, text="0")

        self.activation_increase_label = tk.Label(self.frame, text="Activation increase")
        self.activation_increase_slider = tk.Scale(self.frame, from_=0, to=100, orient=tk.HORIZONTAL)
        self.activation_increase_value = tk.Label(self.frame, text="0")

        self.activation_limit_label = tk.Label(self.frame, text="Activation limit")
        self.activation_limit_slider = tk.Scale(self.frame, from_=0, to=100, orient=tk.HORIZONTAL)
        self.activation_limit_value = tk.Label(self.frame, text="0")

        self.sigmoid_scale_label = tk.Label(self.frame, text="Sigmoid scale")
        self.sigmoid_scale_slider = tk.Scale(self.frame, from_=0, to=100, orient=tk.HORIZONTAL)
        self.sigmoid_scale_value = tk.Label(self.frame, text="0")
        
        self.frame.config(width=400, height=600)
        self.frame.grid()
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(0, weight=1)
        self.frame.pack_propagate(False)
        self.frame.pack()

        self.set_weight_by_gematria_checkbox.grid(row=0, column=0, columnspan=2)
        self.word_change_threshold_label.grid(row=1, column=0)
        self.word_change_threshold_slider.grid(row=1, column=1)
        self.word_change_threshold_value.grid(row=1, column=2)
        self.learning_rate_label.grid(row=2, column=0)
        self.learning_rate_slider.grid(row=2, column=1)
        self.learning_rate_value.grid(row=2, column=2)
        self.error_label.grid(row=3, column=0)
        self.error_slider.grid(row=3, column=1)
        self.error_value.grid(row=3, column=2)
        self.target_label.grid(row=4, column=0)
        self.target_slider.grid(row=4, column=1)
        self.target_value.grid(row=4, column=2)
        self.activation_increase_label.grid(row=5, column=0)
        self.activation_increase_slider.grid(row=5, column=1)
        self.activation_increase_value.grid(row=5, column=2)
        self.activation_limit_label.grid(row=6, column=0)
        self.activation_limit_slider.grid(row=6, column=1)
        self.activation_limit_value.grid(row=6, column=2)
        self.sigmoid_scale_label.grid(row=7, column=0)
        self.sigmoid_scale_slider.grid(row=7, column=1)
        self.sigmoid_scale_value.grid(row=7, column=2)

        self.word_change_threshold_slider.bind("<ButtonRelease-1>", self.updateParams)
        self.learning_rate_slider.bind("<ButtonRelease-1>", self.updateParams)
        self.error_slider.bind("<ButtonRelease-1>", self.updateParams)
        self.target_slider.bind("<ButtonRelease-1>", self.updateParams)
        self.activation_increase_slider.bind("<ButtonRelease-1>", self.updateParams)
        self.activation_limit_slider.bind("<ButtonRelease-1>", self.updateParams)
        self.sigmoid_scale_slider.bind("<ButtonRelease-1>", self.updateParams)

    def updateParams(self, event):
        self.params["word_change_threshold"] = self.word_change_threshold_slider.get()/100
        self.word_change_threshold_value.config(text=str(self.params["word_change_threshold"]))
        self.params["learning_rate"] = self.learning_rate_slider.get()/100
        self.learning_rate_value.config(text=str(self.params["learning_rate"]))
        self.params["error"] = self.error_slider.get()/100
        self.error_value.config(text=str(self.params["error"]))
        self.params["target"] = self.target_slider.get()/100
        self.target_value.config(text=str(self.params["target"]))
        self.params["activation_increase"] = self.activation_increase_slider.get()/1000
        self.activation_increase_value.config(text=str(self.params["activation_increase"]))
        self.params["activation_limit"] = self.activation_limit_slider.get()/100
        self.activation_limit_value.config(text=str(self.params["activation_limit"]))
        self.params["sigmoid_scale"] = self.sigmoid_scale_slider.get()/100
        self.sigmoid_scale_value.config(text=str(self.params["sigmoid_scale"]))

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
        self.words = self.parseText(sys.argv[1])
        self.referenceWords = self.parseText(sys.argv[2])

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
                sentence_gematria = 0
                word_gematria = 0
                for word in sentence.split():
                    word_gematria = get_gematria(word)
                    sentence_gematria += word_gematria
                    self.writeToFile(str(word_gematria) + " + ")
                self.writeToFile(" = " + str(sentence_gematria))
                self.writeToFile(" -> ")
                nr_reduction_array = [sentence_gematria]
                while nr_reduction_array[-1] >= 10:

                    nr_reduction_array.extend([numerological_reduction(sentence_gematria)])
                    sentence_gematria = nr_reduction_array[-1]
                for i in range(len(nr_reduction_array)):
                    self.writeToFile(str(nr_reduction_array[i]))
                    if i < len(nr_reduction_array) - 1:
                        self.writeToFile(" -> ")
                self.writeToFile(" -> ")
                self.writeToFile(str(digital_root(nr_reduction_array[-1])))

                self.writeToFile("\n")
                self.outfile.flush()      

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
    kontrol_thread = threading.Thread(target=kontrol.testNeurons)
    kontrol_thread.start()
    kontrol.tk.mainloop()
    

