import numpy as np
from keras.activations import relu
from scipy.io.wavfile import read, write
from keras.models import Model, Sequential
from keras.layers import Convolution2D, AtrousConvolution2D, Flatten, Dense, Input, Lambda, merge

def wavenetBlock(n_atrous_filters, atrous_filter_size, atrous_rate, n_conv_filters, conv_filter_size):
    def f(input):
        l1 = AtrousConvolution2D(n_atrous_filters, atrous_filter_size, 1, atrous_rate=(atrous_rate, 1), border_mode='same') (input)
        l2a = Convolution2D(n_conv_filters, conv_filter_size, 1, activation='tanh', border_mode='same') (l1)
        l2b = Convolution2D(n_conv_filters, conv_filter_size, 1, activation='sigmoid', border_mode='same') (l1)
        l2 = merge([l2a, l2b], mode='mul')
        l3 = Convolution2D(1, 1, 1, activation='relu', border_mode='same') (l2)
        l4 = merge([l3, input], mode='sum')
        return l4, l3
    return f
        

def get_basic_generative_model()
    input = Input(shape=(1, 4096, 1))
    l1a, l1b = wavenetBlock(16, 2, 2, 5, 3) (input)
    l2a, l2b = wavenetBlock(16, 2, 4, 5, 3) (l1a)
    l3a, l3b = wavenetBlock(16, 2, 8, 5, 3) (l2a)
    l4 = merge([l1b, l2b, l3b], mode='sum')
    l5 = Lambda(relu)(l4)
    l6 = Convolution2D(1, 1, 1, activation='relu') (l5)
    l7 = Convolution2D(1, 1, 1) (l6)
    l8 = Flatten()(l6)
    l9 = Dense(256, activation='softmax') (l8)
    model = Model(input=input, output=l9)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.summary()
    return model

if __name__ == '__main__':
    sr, audio = read('sample.wav')
    audio = audio.astype(float)
    audio = audio - audio.min()
    audio = audio / (audio.max() - audio.min())
    audio = (audio - 0.5) * 2
    X = []
    y = []
    for i in range(0, len(audio)-1, 100):
        frame = audio[i:i+4096]
        if len(frame) < 4096:
            break
        X.append(frame)
        temp = audio[i+4097]
        target_val = int((np.sign(temp) * (np.log(1 + 256*abs(temp)) / (np.log(1+256))) + 1)/2.0 * 255)
        y.append(np.eye(256)[target_val])

    n_examples = len(X)
    X = np.array(X).reshape(n_examples, 1, 4096, 1)
    y = np.array(y)
    model = get_basic_generative_model()
    model.fit(X, y, verbose=1, nb_epoch=20, batch_size=64, validation_split=0.1)
    model.save('my_model.h5')
    # sample from model
    initial_sample = np.zeros((4096)).reshape(1, 1, 4096, 1)
    new_audio = np.zeros((sr * 5)) # generate 5 seconds worth of audio
    curr_sample_idx = 4096
    while curr_sample_idx < new_audio.shape[0]:
        predicted_val = np.argmax(model.predict(initial_sample).reshape(256)))
        ampl_val_8 = ((((predicted_val) / 255.0) - 0.5) * 2.0)
        ampl_val_16 = (np.sign(ampl_val_8) * (1/256.0) * ((1 + 256.0)**abs(ampl_val_8) - 1)) * 2**15
        curr_sample_idx += 1
        new_audio[curr_sample_idx] = ampl_val_16
    write('generated.wav', sr, new_audio.astype(np.int16)) 
