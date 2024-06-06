//using python/cpp interface

#include <iostream>
#include <string>
#include <vector>
#include <alsa/asoundlib.h>
#include <math.h>
#include <time.h>
#include "sanasyna.h"

using namespace std;

// alsa
extern snd_pcm_t *handle;
extern snd_pcm_hw_params_t *params;
extern snd_pcm_uframes_t frames;
extern int dir;
extern snd_pcm_uframes_t period_size;
extern snd_pcm_uframes_t buffer_size;

unsigned int sample_rate = 44100;
double amplitude = 0.0;
double freq = 440.0;


int init_alsa(){
    int rc;
    if((rc = snd_pcm_open(&handle, "default", SND_PCM_STREAM_PLAYBACK, 0)) < 0){
        fprintf(stderr, "unable to open pcm device: %s\n", snd_strerror(rc));
        return 1;
    }

    snd_pcm_hw_params_alloca(&params);

    if((rc = snd_pcm_hw_params_any(handle, params)) < 0){
        fprintf(stderr, "unable to initialize hw parameters: %s\n", snd_strerror(rc));
        return 1;
    }

    if((rc = snd_pcm_hw_params_set_access(handle, params, SND_PCM_ACCESS_RW_INTERLEAVED)) < 0){
        fprintf(stderr, "unable to set access type: %s\n", snd_strerror(rc));
        return 1;
    }

    if((rc = snd_pcm_hw_params_set_format(handle, params, SND_PCM_FORMAT_S16_LE)) < 0){
        fprintf(stderr, "unable to set format: %s\n", snd_strerror(rc));
        return 1;
    }

    if((rc = snd_pcm_hw_params_set_channels(handle, params, 2)) < 0){
        fprintf(stderr, "unable to set channels: %s\n", snd_strerror(rc));
        return 1;
    }

    if((rc = snd_pcm_hw_params_set_rate_near(handle, params, &sample_rate, 0)) < 0){
        fprintf(stderr, "unable to set sample rate: %s\n", snd_strerror(rc));
        return 1;
    }

    if((rc = snd_pcm_hw_params(handle, params)) < 0){
        fprintf(stderr, "unable to set hw parameters: %s\n", snd_strerror(rc));
        return 1;
    }

    snd_pcm_hw_params_get_period_size(params, &period_size, &dir);
    snd_pcm_hw_params_get_buffer_size(params, &buffer_size);

    return 0;
}

int close_alsa(){
    snd_pcm_drain(handle);
    snd_pcm_close(handle);
    return 0;
}

void play(){
    int i;
    int16_t buf[period_size];
    double phase = 0.0;
    double phase_incr = 2.0 * M_PI * freq / sample_rate;
    for(i = 0; i < period_size; i++){
        buf[i] = amplitude * sin(phase);
        phase += phase_incr;
    }
    snd_pcm_writei(handle, buf, period_size);
}

void stop(){
    snd_pcm_drop(handle);
}

void init_sanasyna(){
    init_alsa();
    play();
    stop();
    close_alsa();
}



int main(){
    init_sanasyna();
    return 0;
}

// Compare this snippet from sanasyna.py:

