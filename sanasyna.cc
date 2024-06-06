//using python/cpp interface

#include <iostream>
#include <string>
#include <vector>
#include <jack/jack.h>
#include <math.h>
#include <time.h>
#include "sanasyna.h"

using namespace std;

double amplitude = 0.0;
double freq = 440.0;
int sample_rate = 44100;
int buffer_size = 1024;
jack_client_t *client;
jack_port_t *input_port;
jack_port_t *output_port;
jack_default_audio_sample_t *input_buffer;
jack_default_audio_sample_t *output_buffer;

int process(jack_nframes_t nframes, void *arg){
    jack_default_audio_sample_t *in = (jack_default_audio_sample_t *)jack_port_get_buffer(input_port, nframes);
    jack_default_audio_sample_t *out = (jack_default_audio_sample_t *)jack_port_get_buffer(output_port, nframes);
    for(int i = 0; i < nframes; i++){
        amplitude = sin(2 * M_PI * freq * i / sample_rate);
        out[i] = amplitude * in[i];
    }

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
void init_sanasyna(){
    const char *client_name = "sanasyna";
    jack_options_t options = JackNullOption;
    jack_status_t status;

    client = jack_client_open(client_name, options, &status);
    if(client == NULL){
        std::cout << "Error: jack_client_open() failed, status = " << status << std::endl;
        if(status & JackServerFailed){
            std::cout << "Unable to connect to JACK server" << std::endl;
        }
        exit(1);
    }

    jack_set_process_callback(client, process, 0);
    jack_set_sample_rate_callback(client, srate, 0);
    jack_set_buffer_size_callback(client, buffer, 0);
    jack_on_shutdown(client, jack_shutdown, 0);

    input_port = jack_port_register(client, "input", JACK_DEFAULT_AUDIO_TYPE, JackPortIsInput, 0);
    output_port = jack_port_register(client, "output", JACK_DEFAULT_AUDIO_TYPE, JackPortIsOutput, 0);

    if((input_port == NULL) || (output_port == NULL)){
        std::cout << "Error: jack_port_register() failed" << std::endl;
        exit(1);
    }

    if(jack_activate(client)){
        std::cout << "Error: jack_activate() failed" << std::endl;
        exit(1);
    }

    while(true){
        continue;
    }
}

int main(){
    init_sanasyna();
    return 0;
}

// Compare this snippet from sanasyna.py:

