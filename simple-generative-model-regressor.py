import sys
import time
import numpy as np
from keras.activations import relu
from scipy.io.wavfile import read, write
from keras.models import Model, Sequential
from keras.layers import Convolution2D, AtrousConvolution2D, Flatten, Dense, \
    Input, Lambda, merge


def wavenetBlock(n_atrous_filters, atrous_filter_size, atrous_rate,
                 n_conv_filters, conv_filter_size):
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
    input = Input(shape=(1, input_size, 1))
    l1a, l1b = wavenetBlock(10, 5, 2, 1, 3)(input)
    l2a, l2b = wavenetBlock(1, 2, 4, 1, 3)(l1a)
    l3a, l3b = wavenetBlock(1, 2, 8, 1, 3)(l2a)
    l4a, l4b = wavenetBlock(1, 2, 16, 1, 3)(l3a)
    l5a, l5b = wavenetBlock(1, 2, 32, 1, 3)(l4a)
    l6 = merge([l1b, l2b, l3b, l4b, l5b], mode='sum')
    l7 = Lambda(relu)(l6)
    l8 = Convolution2D(1, 1, 1, activation='relu')(l7)
    l9 = Convolution2D(1, 1, 1)(l8)
    l10 = Flatten()(l9)
    l11 = Dense(1, activation='tanh')(l10)
    model = Model(input=input, output=l11)
    model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
    model.summary()
    return model


def get_audio(filename):
    sr, audio = read(filename)
    audio = audio.astype(float)
    audio = audio - audio.min()
    audio = audio / (audio.max() - audio.min())
    audio = (audio - 0.5) * 2
    return sr, audio


def frame_generator(sr, audio, frame_size, frame_shift):
    audio_len = len(audio)
    while 1:
        for i in range(0, audio_len - frame_size - 1, frame_shift):
            frame = audio[i:i+frame_size]
            if len(frame) < frame_size:
                break
            if i + frame_size >= audio_len:
                break
            temp = audio[i + frame_size]
            yield frame.reshape(1, 1, frame_size, 1), \
                temp.reshape(1, 1)


if __name__ == '__main__':
    n_epochs = 20
    frame_size = 2048
    frame_shift = 512
    sr_training, training_audio = get_audio('train.wav')
    training_audio = training_audio[:sr_training*240]
    sr_valid, valid_audio = get_audio('validate.wav')
    valid_audio = valid_audio[:sr_valid*30]
    assert sr_training == sr_valid, "Training, validation samplerate mismatch"
    n_training_examples = int((len(training_audio)-frame_size-1) / float(
        frame_shift))
    n_validation_examples = int((len(valid_audio)-frame_size-1) / float(
        frame_shift))
    model = get_basic_generative_model(frame_size)
    print 'Total training examples:', n_training_examples
    print 'Total validation examples:', n_validation_examples
    model.fit_generator(frame_generator(sr_training, training_audio,
                                        frame_size, frame_shift),
                        samples_per_epoch=n_training_examples,
                        nb_epoch=n_epochs,
                        validation_data=frame_generator(sr_valid, valid_audio,
                                                        frame_size, frame_shift
                                                        ),
                        nb_val_samples=n_validation_examples,
                        verbose=1)
    print 'Saving model...'
    str_timestamp = str(int(time.time()))
    model.save('models/model_'+str_timestamp+'_'+str(n_epochs)+'.h5')
    print 'Generating audio...'
    new_audio = np.zeros((sr_training * 3))
    curr_sample_idx = 0
    audio_context = valid_audio[:frame_size]
    while curr_sample_idx < new_audio.shape[0]:
        predicted_val = model.predict(audio_context.reshape(1, 1, frame_size,
                                                            1))
        ampl_val_16 = predicted_val * 2**15
        new_audio[curr_sample_idx] = ampl_val_16
        audio_context[-1] = ampl_val_16
        audio_context[:-1] = audio_context[1:]
        pc_str = str(round(100*curr_sample_idx/float(new_audio.shape[0]), 2))
        sys.stdout.write('Percent complete: ' + pc_str + '\r')
        sys.stdout.flush()
        curr_sample_idx += 1
    outfilepath = 'output/reg_generated_'+str_timestamp+'.wav'
    print 'Writing generated audio to:', outfilepath
    write(outfilepath, sr_training, new_audio.astype(np.int16))
    print '\nDone!'
