#include "sanasyna.h"
#include <SFML/Audio.hpp>
#include <Python.h>
#include <iostream>
#include <pybind11/pybind11.h>

using namespace std;
using namespace pybind11;

// sfml
sf::SoundBuffer audio_buffer;
sf::Sound sound;

void play() {
    sound.play();
}

void stop() {
    sound.stop();
}

void close() {
    sound.stop();
}

void init_audio() {
    sound.setBuffer(audio_buffer);
}

double amplitude;
double freq;
int sample_rate;

int init_sanasyna() {
    return 0;
}

void generate_sine_wave(double freq, double amplitude, int sample_rate) {
    // Generate a sine wave
    int num_samples = sample_rate;
    sf::Int16 *samples = new sf::Int16[num_samples];
    for (int i = 0; i < num_samples; i++) {
        samples[i] = amplitude * sin(2 * M_PI * freq * i / sample_rate);
    }
    audio_buffer.loadFromSamples(samples, num_samples, 1, sample_rate);
    delete[] samples;
}

void generate_square_wave(double freq, double amplitude, int sample_rate) {
    // Generate a square wave
    int num_samples = sample_rate;
    sf::Int16 *samples = new sf::Int16[num_samples];
    for (int i = 0; i < num_samples; i++) {
        samples[i] = amplitude * (sin(2 * M_PI * freq * i / sample_rate) > 0 ? 1 : -1);
    }
    audio_buffer.loadFromSamples(samples, num_samples, 1, sample_rate);
    delete[] samples;
}

void generate_sawtooth_wave(double freq, double amplitude, int sample_rate) {
    // Generate a sawtooth wave
    int num_samples = sample_rate;
    sf::Int16 *samples = new sf::Int16[num_samples];
    for (int i = 0; i < num_samples; i++) {
        samples[i] = (int)amplitude * (2 * (i % ((int)sample_rate / (int)freq)) / ((int)sample_rate / (int)freq) - 1);
    }
    audio_buffer.loadFromSamples(samples, num_samples, 1, sample_rate);
    delete[] samples;
}

void generate_triangle_wave(double freq, double amplitude, int sample_rate) {
    // Generate a triangle wave
    int num_samples = sample_rate;
    sf::Int16 *samples = new sf::Int16[num_samples];
    for (int i = 0; i < num_samples; i++) {
        samples[i] = amplitude * (2 * abs(2 * (i % ((int)sample_rate / (int)freq)) / ((int)sample_rate / (int)freq) - 1) - 1);
    }
    audio_buffer.loadFromSamples(samples, num_samples, 1, sample_rate);
    delete[] samples;
}

void generate_noise_wave(double freq, double amplitude, int sample_rate) {
    // Generate a noise wave
    int num_samples = sample_rate;
    sf::Int16 *samples = new sf::Int16[num_samples];
    for (int i = 0; i < num_samples; i++) {
        samples[i] = amplitude * (rand() % 32768 - 16384);
    }
    audio_buffer.loadFromSamples(samples, num_samples, 1, sample_rate);
    delete[] samples;
}

void generate_melody(int *melody, double amplitude, int sample_rate) {
    // Generate a melody
    int num_samples = sample_rate;
    sf::Int16 *samples = new sf::Int16[num_samples];
    for (int i = 0; i < num_samples; i++) {
        samples[i] = amplitude * sin(2 * M_PI * melody[i % 8] * i / sample_rate);
    }
    audio_buffer.loadFromSamples(samples, num_samples, 1, sample_rate);
    delete[] samples;
}

void set_amplitude(double amplitude) {
    amplitude = amplitude;
}

void set_freq(double freq) {
    freq = freq;
}

void set_sample_rate(int sample_rate) {
    sample_rate = sample_rate;
}

void set_buffer(sf::Int16 *samples, int sample_rate) {
    audio_buffer.loadFromSamples(samples, sample_rate, 1, sample_rate);
}

void set_sound_buffer(sf::SoundBuffer buffer) {
    audio_buffer = buffer;
}

void set_buffer_from_samples(sf::Int16 *samples, int sample_rate, int channels) {
    audio_buffer.loadFromSamples(samples, sample_rate, channels, sample_rate);
}

void set_buffer_from_file(const char *filename) {
    audio_buffer.loadFromFile(filename);
}

void set_sound_buffer_from_file(const char *filename) {
    audio_buffer.loadFromFile(filename);
}

// module definition

PyModuleDef *sanasyna = {
    PyModuleDef_HEAD_INIT,
    "sanasyna",
    "sanasyna module",
    -1,
    NULL, NULL, NULL, NULL, NULL
};


PyAPI_FUNC(PyObject *) PyModule_Create2(&PyModuleDef, int apiver) {
    return PyModule_Create(&sanasyna);
}

// init routine
//PyObject *sanasyna = PyInit__sanasyna();

// bind to python
PYBIND11_MODULE(sanasyna, m) {
    m.def("play", &play);
    m.def("stop", &stop);
    //m.def("close", &close);
    m.def("init_audio", &init_audio);
    m.def("init_sanasyna", &init_sanasyna);
    m.def("generate_sine_wave", &generate_sine_wave);
    m.def("generate_square_wave", &generate_square_wave);
    m.def("generate_sawtooth_wave", &generate_sawtooth_wave);
    m.def("generate_triangle_wave", &generate_triangle_wave);
    m.def("generate_noise_wave", &generate_noise_wave);
    m.def("generate_melody", &generate_melody);
    m.def("set_amplitude", &set_amplitude);
    m.def("set_freq", &set_freq);
    m.def("set_sample_rate", &set_sample_rate);
    m.def("set_buffer", &set_buffer);
    m.def("set_sound_buffer", &set_sound_buffer);
    m.def("set_buffer_from_samples", &set_buffer_from_samples);
    m.def("set_buffer_from_file", &set_buffer_from_file);
    m.def("set_sound_buffer_from_file", &set_sound_buffer_from_file);
}

// End of sanasyna.cc
