// Purpose: SWIG interface file for sanasyna module.


// Define the module name
%module sanasyna

%{
    #include "sanasyna.h"
    #include <asoundlib.h>

    extern double amplitude;
    extern double freq;

    // alsa
    extern snd_pcm_t *handle;
    extern snd_pcm_hw_params_t *params;
    extern snd_pcm_uframes_t frames;
    extern int dir;
    extern snd_pcm_uframes_t period_size;
    extern snd_pcm_uframes_t buffer_size;

    int init_alsa();
    void play();
    void stop();
    void close();

%}
