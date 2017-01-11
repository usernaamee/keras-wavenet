import os
import sys
import time
import numpy as np
from keras.callbacks import Callback
from scipy.io.wavfile import read, write
from keras.models import Model, Sequential
from keras.layers import Convolution1D, AtrousConvolution1D, Flatten, Dense, \
    Input, Lambda, merge, Activation


def wavenetBlock(n_atrous_filters, atrous_filter_size, atrous_rate):
    def f(input_):
        residual = input_
        tanh_out = AtrousConvolution1D(n_atrous_filters, atrous_filter_size,
                                       atrous_rate=atrous_rate,
                                       border_mode='same',
                                       activation='tanh')(input_)
        sigmoid_out = AtrousConvolution1D(n_atrous_filters, atrous_filter_size,
                                          atrous_rate=atrous_rate,
                                          border_mode='same',
                                          activation='sigmoid')(input_)
        merged = merge([tanh_out, sigmoid_out], mode='mul')
        skip_out = Convolution1D(1, 1, activation='relu', border_mode='same')(merged)
        out = merge([skip_out, residual], mode='sum')
        return out, skip_out
    return f


def get_basic_generative_model(input_size):
    input_ = Input(shape=(input_size, 1))
    A, B = wavenetBlock(64, 2, 2)(input_)
    skip_connections = [B]
    for i in range(20):
        A, B = wavenetBlock(64, 2, 2**((i+2)%9))(A)
        skip_connections.append(B)
    net = merge(skip_connections, mode='sum')
    net = Activation('relu')(net)
    net = Convolution1D(1, 1, activation='relu')(net)
    net = Convolution1D(1, 1)(net)
    net = Flatten()(net)
    net = Dense(256, activation='softmax')(net)
    model = Model(input=input_, output=net)
    model.compile(loss='categorical_crossentropy', optimizer='sgd',
                  metrics=['accuracy'])
    model.summary()
    return model


def get_audio(filename):
    sr, audio = read(filename)
    audio = audio.astype(float)
    audio = audio - audio.min()
    audio = audio / (audio.max() - audio.min())
    audio = (audio - 0.5) * 2
    return sr, audio


def frame_generator(sr, audio, frame_size, frame_shift, minibatch_size=20):
    audio_len = len(audio)
    X = []
    y = []
    while 1:
        for i in range(0, audio_len - frame_size - 1, frame_shift):
            frame = audio[i:i+frame_size]
            if len(frame) < frame_size:
                break
            if i + frame_size >= audio_len:
                break
            temp = audio[i + frame_size]
            target_val = int((np.sign(temp) * (np.log(1 + 256*abs(temp)) / (
                np.log(1+256))) + 1)/2.0 * 255)
            X.append(frame.reshape(frame_size, 1))
            y.append((np.eye(256)[target_val]))
            if len(X) == minibatch_size:
                yield np.array(X), np.array(y)
                X = []
                y = []


def get_audio_from_model(model, sr, duration, seed_audio):
    print 'Generating audio...'
    new_audio = np.zeros((sr * duration))
    curr_sample_idx = 0
    while curr_sample_idx < new_audio.shape[0]:
        distribution = np.array(model.predict(seed_audio.reshape(1,
                                                                 frame_size, 1)
                                             ), dtype=float).reshape(256)
        distribution /= distribution.sum().astype(float)
        predicted_val = np.random.choice(range(256), p=distribution)
        ampl_val_8 = ((((predicted_val) / 255.0) - 0.5) * 2.0)
        ampl_val_16 = (np.sign(ampl_val_8) * (1/256.0) * ((1 + 256.0)**abs(
            ampl_val_8) - 1)) * 2**15
        new_audio[curr_sample_idx] = ampl_val_16
        seed_audio[-1] = ampl_val_16
        seed_audio[:-1] = seed_audio[1:]
        pc_str = str(round(100*curr_sample_idx/float(new_audio.shape[0]), 2))
        sys.stdout.write('Percent complete: ' + pc_str + '\r')
        sys.stdout.flush()
        curr_sample_idx += 1
    print 'Audio generated.'
    return new_audio.astype(np.int16)


class SaveAudioCallback(Callback):
    def __init__(self, ckpt_freq, sr, seed_audio):
        super(SaveAudioCallback, self).__init__()
        self.ckpt_freq = ckpt_freq
        self.sr = sr
        self.seed_audio = seed_audio

    def on_epoch_end(self, epoch, logs={}):
        if (epoch+1)%self.ckpt_freq==0:
            ts = str(int(time.time()))
            filepath = os.path.join('output/', 'ckpt_'+ts+'.wav')
            audio = get_audio_from_model(self.model, self.sr, 0.5, self.seed_audio)
            write(filepath, self.sr, audio)


if __name__ == '__main__':
    n_epochs = 2000
    frame_size = 2048
    frame_shift = 128
    sr_training, training_audio = get_audio('train.wav')
    # training_audio = training_audio[:sr_training*1200]
    sr_valid, valid_audio = get_audio('validate.wav')
    # valid_audio = valid_audio[:sr_valid*60]
    assert sr_training == sr_valid, "Training, validation samplerate mismatch"
    n_training_examples = int((len(training_audio)-frame_size-1) / float(
        frame_shift))
    n_validation_examples = int((len(valid_audio)-frame_size-1) / float(
        frame_shift))
    model = get_basic_generative_model(frame_size)
    print 'Total training examples:', n_training_examples
    print 'Total validation examples:', n_validation_examples
    audio_context = valid_audio[:frame_size]
    save_audio_clbk = SaveAudioCallback(100, sr_training, audio_context)
    validation_data_gen = frame_generator(sr_valid, valid_audio, frame_size, frame_shift)
    training_data_gen = frame_generator(sr_training, training_audio, frame_size, frame_shift)
    model.fit_generator(training_data_gen, samples_per_epoch=3000, nb_epoch=n_epochs, validation_data=validation_data_gen,nb_val_samples=500, verbose=1, callbacks=[save_audio_clbk])
    print 'Saving model...'
    str_timestamp = str(int(time.time()))
    model.save('models/model_'+str_timestamp+'_'+str(n_epochs)+'.h5')
    print 'Generating audio...'
    new_audio = get_audio_from_model(model, sr_training, 2, audio_context)
    outfilepath = 'output/generated_'+str_timestamp+'.wav'
    print 'Writing generated audio to:', outfilepath
    write(outfilepath, sr_training, new_audio)
    print '\nDone!'
