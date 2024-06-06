#include <jack/jack.h>

extern double amplitude;
extern double freq;
extern int sample_rate;
extern int buffer_size;
extern jack_client_t *client;
extern jack_port_t *input_port;
extern jack_port_t *output_port;
extern jack_default_audio_sample_t *input_buffer;
extern jack_default_audio_sample_t *output_buffer;

int process(jack_nframes_t nframes, void *arg);
int srate(jack_nframes_t nframes, void *arg);
int buffer(jack_nframes_t nframes, void *arg);
void error(const char *desc);
void jack_shutdown(void *arg);
void init_sanasyna();
int main();

//
//
// Compare this snippet from sanasyna.py: