import pickle
import librosa
import pyaudio
import numpy as np
import sys
import sounddevice as sd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from PyQt5 import  QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow

CHUNK = 2048
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 2
seconds = 1

scaler = StandardScaler()
scaler_filename = "E:\\GoogleDrive\\MestradoEEC\\LabInt_2\\MEEC_1920_LI2_G1\\Treino\\sc_model_person.bin"
scaler = pickle.load(open(scaler_filename,"rb"))

scaler_cmd = StandardScaler()
scaler_cmd_filename = "E:\\GoogleDrive\\MestradoEEC\\LabInt_2\\MEEC_1920_LI2_G1\\Treino\\sc_model_command.bin"
scaler_cmd = pickle.load(open(scaler_cmd_filename,"rb"))

mlp = MLPClassifier()
filename = "E:\\GoogleDrive\\MestradoEEC\\LabInt_2\\MEEC_1920_LI2_G1\\Treino\\trained_model_person.bin"
mlp = pickle.load(open(filename,"rb"))

mlp_cmd = MLPClassifier()
filename_cmd = "E:\\GoogleDrive\\MestradoEEC\\LabInt_2\\MEEC_1920_LI2_G1\\Treino\\trained_model_command.bin"
mlp_cmd = pickle.load(open(filename_cmd,"rb"))

grupo = ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8")
command = ("parar", "recuar", "direita", "esquerda", "baixo", "centro", "cima", "avançar")

class MyWindow(QMainWindow):
    def __init__(self):
        super(MyWindow,self).__init__()
        self.initUI()

    def button_clicked(self):
        clip_audio = get_audio()
        X_person = prep_data(clip_audio)
        X_cmd = prep_data_cmd(clip_audio)
        result, result_cmd = prev_result(X_person, X_cmd)

        self.user.setText(result)
        self.comando.setText(result_cmd)
        print(result)
        print(result_cmd)

    def initUI(self):
        self.setGeometry(100, 100, 800, 600)
        self.setWindowTitle("LI Grupo 1")

        self.label = QtWidgets.QLabel(self)
        self.label.setText("Utilizador: ")
        self.label.move(150,550)
        self.user = QtWidgets.QLabel(self)
        self.user.setText(" ")
        self.user.move(250, 550)

        self.label_2 = QtWidgets.QLabel(self)
        self.label_2.setText("Comando: ")
        self.label_2.move(400, 550)
        self.comando = QtWidgets.QLabel(self)
        self.comando.setText(" ")
        self.comando.move(500, 550)

        self.b1 = QtWidgets.QPushButton(self)
        self.b1.setText("Iniciar")
        self.b1.clicked.connect(self.button_clicked)
        self.b1.move(10, 550)

    def update(self):
        self.label.adjustSize()

def window():
    app = QApplication(sys.argv)
    win = MyWindow()
    win.show()
    sys.exit(app.exec_())

def get_audio ():

    signal, sample_rate = librosa.load("E:\GoogleDrive\MestradoEEC\LabInt_2\MEEC_1920_LI2_G1\AudioDatasetRec\G1\G1_Recuar_12.wav", sr=RATE)
    # p = pyaudio.PyAudio()
    #
    # stream = p.open(format=FORMAT,
    #                 channels=CHANNELS,
    #                 rate=RATE,
    #                 input=True,
    #                 frames_per_buffer=CHUNK)
    #
    # print("* recording")
    #
    # frames = []
    #
    # for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    #     data = stream.read(CHUNK)
    #     frames.append(data)
    #
    # print("* done recording")

    # stream.stop_stream()
    # stream.close()
    # p.terminate()

    # print("A detetar!")
    # frames = sd.rec(int(seconds * RATE), samplerate=RATE, channels=2)
    # sd.wait()
    # frames = librosa.to_mono(frames)
    # print("A processar!")

    return signal


def prep_data (clip_audio):
    mfcc = np.mean(librosa.feature.mfcc(clip_audio, sr=RATE, n_mfcc=13, n_fft=2048, hop_length=512).T,axis=0)
    X = []
    X.append(mfcc)
    X_person = scaler.transform(X)

    return X_person

def prev_result (X_person, X_cmd):
    predictions = mlp.predict(X_person)
    predictions_cmd = mlp_cmd.predict(X_cmd)

    result = grupo[int(predictions)]
    result_cmd = command[int(predictions_cmd)]
    return result, result_cmd


def prep_data_cmd(clip_audio):
    W = []
    result = np.array([])

    mfcc = np.mean(librosa.feature.mfcc(clip_audio, sr=RATE, n_mfcc=13, n_fft=2048, hop_length=512).T, axis=0)
    result = np.hstack((result, mfcc))
    stft = np.abs(librosa.stft(clip_audio))
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=RATE).T, axis=0)
    result = np.hstack((result, chroma))
    mel = np.mean(librosa.feature.melspectrogram(clip_audio, sr=RATE).T, axis=0)
    result = np.hstack((result, mel))
    W.append(result)

    X_cmd = scaler_cmd.transform(W)
    return X_cmd


if __name__ == "__main__":
    window()
    # clip_audio = get_audio()
    # X_person = prep_data(clip_audio)
    # X_cmd = prep_data_cmd(clip_audio)
    #
    # result, result_cmd = prev_result(X_person, X_cmd)
    # print(result)
    # print(result_cmd)

