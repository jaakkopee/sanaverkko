//using python/cpp interface

#include <iostream>
#include <string>
#include <vector>
#include <jack/jack.h>

extern double amplitude;
extern int sample_rate;
extern int buffer_size;
extern jack_client_t *client;
extern jack_port_t *input_port;
extern jack_port_t *output_port;
extern jack_default_audio_sample_t *input_buffer;
extern jack_default_audio_sample_t *output_buffer;

int process(jack_nframes_t nframes, void *arg){
    std::cout << "Processing" << std::endl;
    return 0;
}

int srate(jack_nframes_t nframes, void *arg){
    std::cout << "Setting sample rate" << std::endl;
    return 0;
}

int buffer(jack_nframes_t nframes, void *arg){
    std::cout << "Setting buffer size" << std::endl;
    return 0;
}

void error(const char *desc){
    std::cout << "Error: " << desc << std::endl;
}

void jack_shutdown(void *arg){
    std::cout << "Jack shutdown" << std::endl;
}

// Compare this snippet from sanasyna.py:

