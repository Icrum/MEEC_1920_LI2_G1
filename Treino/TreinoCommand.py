import numpy as np
import os
import librosa
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from Treino.AudioAugmentation import AudioAugmentation


DATASET_PATH = "E:\GoogleDrive\MestradoEEC\LabInt_2\MEEC_1920_LI2_G1\DatasetTreinoCommand"
SAMPLE_RATE = 44100
command = []

Naug = 11 # Maximo 11
Segundos = 2.3
aa = AudioAugmentation()

def save_mfcc(dataset_path, num_mfcc=13, n_fft=2048, hop_length=512, adicaoControlo = 80000):
    X=[]
    y=[]

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        semantic_label = dirpath.split("\\")[-1]
        command.append(semantic_label)
        for f in filenames:
            #result = np.array([])
            file_path = os.path.join(dirpath, f)
            # signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
            # mfcc = np.mean(librosa.feature.mfcc(signal, sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length).T, axis=0)
            # result = np.hstack((result, mfcc))
            # stft = np.abs(librosa.stft(signal))
            # chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            # result = np.hstack((result, chroma))
            # mel = np.mean(librosa.feature.melspectrogram(signal, sr=sample_rate).T, axis=0)
            # result = np.hstack((result, mel))
            #
            # X.append(result)
            # y.append(i - 1)
            # print("{}".format(file_path))

            data, adicaoMin = aa.read_audio_file(file_path, SAMPLE_RATE, Segundos)
            a = 0
            while True:
                # Adding noise to sound
                result = np.array([])
                data_shift = aa.shift(data, a)
                mfcc = np.mean(librosa.feature.mfcc(data_shift, sr=SAMPLE_RATE, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length).T, axis=0)
                result = np.hstack((result, mfcc))
                # stft = np.abs(librosa.stft(data_shift))
                # chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=SAMPLE_RATE).T, axis=0)
                # result = np.hstack((result, chroma))
                # mel = np.mean(librosa.feature.melspectrogram(data_shift, sr=SAMPLE_RATE).T, axis=0)
                # result = np.hstack((result, mel))

                X.append(result)
                y.append(i - 1)

                # print("{}".format(file_path))

                if a == Naug:
                    break
                else:
                    a += 1

            # result = np.array([])
            # data_noise = aa.add_noise(data, a)
            # mfcc = np.mean(
            #     librosa.feature.mfcc(data_noise, sr=SAMPLE_RATE, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length).T,
            #     axis=0)
            # result = np.hstack((result, mfcc))
            # stft = np.abs(librosa.stft(data_noise))
            # chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=SAMPLE_RATE).T, axis=0)
            # result = np.hstack((result, chroma))
            # mel = np.mean(librosa.feature.melspectrogram(data_noise, sr=SAMPLE_RATE).T, axis=0)
            # result = np.hstack((result, mel))

            # X.append(result)
            # y.append(i - 1)
            print("Numero de repeticoes: ", a)
            print("{}".format(file_path))

            if adicaoMin < adicaoControlo:
                adicaoControlo = adicaoMin

    print("Adição minima: ", adicaoControlo)
    return X, y


if __name__ == "__main__":

    X,y = save_mfcc(DATASET_PATH)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7)

    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    sc_filename = 'sc_model_command.bin'
    pickle.dump(scaler, open(sc_filename, 'wb'))

    mlp = MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
                        beta_2=0.999, early_stopping=False, epsilon=1e-08,
                        hidden_layer_sizes=(52, 26, 13), learning_rate='constant',
                        learning_rate_init=0.001, max_iter=200, momentum=0.9,
                        nesterovs_momentum=True, power_t=0.5, random_state=7,
                        shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
                        verbose=False, warm_start=False)
    mlp.fit(X_train, y_train)

    predictions = mlp.predict(X_test)

    filename = 'trained_model_command.bin'
    pickle.dump(mlp, open(filename, 'wb'))

    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))
    print(command)

