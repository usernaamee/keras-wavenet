# keras-wavenet
Keras implementation of deepmind's wavenet paper

[Link to paper](https://drive.google.com/file/d/0B3cxcnOkPx9AeWpLVXhkTDJINDQ/view)

## Dataset used
I have used [Librispeech](http://www.openslr.org/12/) corpus. I have concatenated all audio files in dev-clean to create train.wav and all files in test-clean to create validate.wav. I have resampled the audio files to 8000 Hz.
Here is how you can create train.wav & validate.wav using vlc on linux:  
```
cvlc -vvv --sout-keep --sout-all --sout "#gather:transcode{acodec=s16l,channels=1,samplerate=8000}:std{access=file,mux=wav,dst=validate.wav}" `find LibriSpeech/test-clean/ -name "*.flac"` vlc://quit
cvlc -vvv --sout-keep --sout-all --sout "#gather:transcode{acodec=s16l,channels=1,samplerate=8000}:std{access=file,mux=wav,dst=train.wav}" `find LibriSpeech/dev-clean/ -name "*.flac"` vlc://quit
```

## Todo
- [x] The basic generative model  
- [ ] Conditioning logic (speaker)  
- [ ] Conditioning logic (TTS)  
