#include <alsa/asoundlib.h>

// alsa
extern snd_pcm_t *handle;
extern snd_pcm_hw_params_t *params;
extern snd_pcm_uframes_t frames;
extern int dir;
extern snd_pcm_uframes_t period_size;
extern snd_pcm_uframes_t buffer_size;

extern unsigned int sample_rate;
extern double amplitude;
extern double freq;


int init_alsa();
int close_alsa();
void play();
void stop();
void close();

// Path: sanasyna.h


