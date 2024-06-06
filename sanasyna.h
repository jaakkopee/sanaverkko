#include <SFML/Audio.hpp>
#include <Python.h>
#include <iostream>
#include <pybind11/pybind11.h>

using namespace std;
using namespace pybind11;


extern PyObject* PyInit__sanasyna();
extern PyModuleDef *sanasyna;
extern PyAPI_FUNC(PyObject *) PyModule_Create2(&PyModuleDef, int apiver);

// sfml
extern sf::SoundBuffer audio_buffer;
extern sf::Sound sound;

extern void play();
extern void stop();
extern void close();
extern void init_audio();

extern double amplitude;
extern double freq;
extern int sample_rate;

extern int init_sanasyna();
extern void generate_sine_wave(double freq, double amplitude, int sample_rate);
extern void generate_square_wave(double freq, double amplitude, int sample_rate);
extern void generate_sawtooth_wave(double freq, double amplitude, int sample_rate);
extern void generate_triangle_wave(double freq, double amplitude, int sample_rate);
extern void generate_noise_wave(double freq, double amplitude, int sample_rate);
extern void generate_melody(int *melody, double amplitude, int sample_rate);


extern void set_amplitude(double amplitude);
extern void set_freq(double freq);
extern void set_sample_rate(int sample_rate);

extern void set_buffer(sf::Int16 *samples, int sample_rate);
extern void set_sound_buffer(sf::SoundBuffer buffer);

extern void set_buffer_from_samples(sf::Int16 *samples, int sample_rate, int channels);
extern void set_buffer_from_file(const char *filename);
extern void set_sound_buffer_from_file(const char *filename);


// End of sanasyna.h


