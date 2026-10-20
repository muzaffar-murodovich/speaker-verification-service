/* G.729 round trip for the degradation harness: 8 kHz s16le on stdin -> encode
 * -> decode -> 8 kHz s16le on stdout. ffmpeg can decode G.729 but not encode it,
 * so this wraps the bcg729 library (Annex A codec, 8 kbit/s). Built by `make g729`.
 *
 * The codec works on 10 ms frames (80 samples -> 10 bytes). A final partial frame
 * is zero-padded, so the output can be up to 79 samples longer; the caller trims it.
 * The encoder's VAD/DTX is off: every frame is coded, as on a call without silence
 * suppression.
 */
#include <bcg729/decoder.h>
#include <bcg729/encoder.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define FRAME 80

int main(void) {
    bcg729EncoderChannelContextStruct *encoder = initBcg729EncoderChannel(0);
    bcg729DecoderChannelContextStruct *decoder = initBcg729DecoderChannel();
    if (!encoder || !decoder) {
        fprintf(stderr, "g729: cannot initialise the codec\n");
        return 1;
    }
    int16_t in[FRAME], out[FRAME];
    uint8_t bits[10], length;
    size_t got;
    while ((got = fread(in, sizeof(int16_t), FRAME, stdin)) > 0) {
        if (got < FRAME) memset(in + got, 0, (FRAME - got) * sizeof(int16_t));
        bcg729Encoder(encoder, in, bits, &length);
        bcg729Decoder(decoder, bits, length, 0, 0, 0, out);
        if (fwrite(out, sizeof(int16_t), FRAME, stdout) != FRAME) return 1;
    }
    closeBcg729EncoderChannel(encoder);
    closeBcg729DecoderChannel(decoder);
    return ferror(stdin) ? 1 : 0;
}
