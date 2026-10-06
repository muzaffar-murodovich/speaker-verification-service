# D2: audio fundamentals

## Samples and bandwidth

Sample rate is the number of measurements per second: 8 kHz means 8,000 samples
per second. The theoretical highest representable frequency is half that rate.
Telephone narrowband speech often carries roughly 300–3,400 Hz. Resampling an
8 kHz recording to 16 kHz supplies the format a model expects; it cannot restore
missing high-frequency information. Downsampling needs an anti-alias filter so
frequencies above the new limit do not fold into lower frequencies.

## Codecs and clipping

A codec controls how audio is stored or transmitted; it is different from sample
rate. G.711 A-law and mu-law use nonlinear quantisation, commonly at 8 kHz. G.729
compresses narrowband speech more aggressively. Opus can support narrowband and
wideband audio. Compression can change the cues used to compare speakers.
Codec degradation experiments belong to D6; D2 does not implement codec encoders.

Clipping happens when the signal exceeds the representable amplitude: peaks are
cut flat and information is lost. Lowering the volume later cannot undo it.
The VAD input is clipped to the PCM range to avoid integer overflow; the returned
floating-point speech is not volume-normalised or repaired.

## Channels, segments and VAD

A channel is one audio stream. In a two-channel call, each side may contain a
different person. Select a channel explicitly; averaging them would mix voices.
This pipeline does not perform diarization or identify the speaker.

Segments are millisecond intervals [start, end). They specify where to look for
one speaker's speech. VAD (voice activity detection) then classifies short frames
inside those intervals as speech or non-speech. Pooling concatenates only the
frames classified as speech, in chronological order.

Net speech seconds count retained samples, excluding padding. Speech ratio is
net speech divided by the duration of the selected segments, excluding gaps.
No detected speech returns an empty array. Downstream scoring must enforce the
3-second minimum; do not confuse selected duration with net speech duration.

WebRTC VAD is a lightweight baseline, not a guarantee that music, noise or another
speaker will be rejected. Its accuracy on telephony data has not been measured
here. Quality scoring and turn consistency are not implemented in D2.
