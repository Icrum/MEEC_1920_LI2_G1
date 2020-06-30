import pickle
import librosa
import pyaudio
import numpy as np
import sys
import sounddevice as sd
import matplotlib.pyplot as plt
import librosa.display
import time
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

CHUNK = 2048
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 2.3
# seconds = 0.5

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

grupo = ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "Desc")
command = ("parar", "recuar", "direita", "esquerda", "baixo", "centro", "cima", "avançar", "Desc")


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

class AudioHandler(object):
    def __init__(self):
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 44100
        self.CHUNK = 1024 * 2
        self.p = None
        self.stream = None
        self.numpy_array = []
        self.frames = []
        self.result = 8
        self.result_cmd = 8

    def start(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.FORMAT,
                                  channels=self.CHANNELS,
                                  rate=self.RATE,
                                  input=True,
                                  output=False,
                                  stream_callback=self.callback,
                                  frames_per_buffer=self.CHUNK)

    def stop(self):
        self.stream.close()
        self.p.terminate()

    def callback(self, in_data, frame_count, time_info, flag):
        self.numpy_array = np.frombuffer(in_data, dtype=np.float16)
        # librosa.feature.mfcc(numpy_array)
        return None, pyaudio.paContinue

    def mainloop(self):
        while (self.stream.is_active()): # if using button you can set self.stream to 0 (self.stream = 0), otherwise you can use a stop condition

            npSize = len(self.numpy_array)

            if npSize >= RATE*RECORD_SECONDS:
                self.frames.append(self.numpy_array[npSize-(RATE*RECORD_SECONDS):npSize])
                X_person = prep_data(self.frames)
                X_cmd = prep_data_cmd(self.frames)
                result, result_cmd = prev_result(X_person, X_cmd)
            else:
                time.sleep(0.2)

            return self.result, self.result_cmd

class MyWindow(QMainWindow):
    def __init__(self):
        super(MyWindow,self).__init__()
        self.initUI()

    def button_clicked(self):
        audio = AudioHandler()
        audio.start()  # open the the stream
        result, result_cmd = audio.mainloop()  # main operations with librosa
        # audio.stop()

        # clip_audio, sample_rate = get_audio()
        # show_graf(clip_audio, RATE)

        # X_person = prep_data(clip_audio)
        # X_cmd = prep_data_cmd(clip_audio)
        # result, result_cmd = prev_result(X_person, X_cmd)

        self.user.setText(grupo[result])
        self.comando.setText(command[result_cmd])
        print(result)
        print(result_cmd)

    def initUI(self):
        self.setGeometry(500, 100, 800, 600)
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
        
        
def show_graf(clip_audio, sample_rate):
    # sc = MplCanvas(MyWindow, width=5, height=4, dpi=100)
    # sc.axes(clip_audio)
    # MyWindow.setCentralWidget(sc)
    # MyWindow.show()
    pass
    
def window():
    app = QApplication(sys.argv)
    win = MyWindow()
    win.show()
    sys.exit(app.exec_())

def get_audio():
    # Source file
    # signal, sample_rate = librosa.core.load("E:\GoogleDrive\MestradoEEC\LabInt_2\MEEC_1920_LI2_G1\AudioDatasetRec\G1\G1_Esquerda_1.wav", sr=RATE)
    # plt.subplot(211)
    # plt.plot(signal)
    # # plt.xlabel('Tempo')
    # plt.ylabel('Amplitude')
    # plt.subplot(212)
    # powerSpectrum, freqenciesFound, time, imageAxis = plt.specgram(signal, Fs=sample_rate)
    # plt.xlabel('Tempo')
    # plt.ylabel('Frequência')
    # plt.show()

    #Source stream
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("* recording")

    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    print("A detetar!")
    # signal = sd.RawStream(device=None, samplerate=RATE, channels=1, blocksize=int(RATE/CHUNK * RECORD_SECONDS))
    # signal = sd.rec(int(RECORD_SECONDS * RATE), samplerate=RATE, channels=2)
    # sd.wait()
    # signal, sample_rate = librosa.core.load(frames)
    signal = librosa.to_mono(frames)
    print("A processar!")
    plt.subplot(211)
    plt.plot(signal)
    # plt.xlabel('Tempo')
    plt.ylabel('Amplitude')
    plt.subplot(212)
    powerSpectrum, freqenciesFound, time, imageAxis = plt.specgram(signal, Fs=RATE)
    plt.xlabel('Tempo')
    plt.ylabel('Frequência')
    plt.show()

    return signal, RATE


def prep_data (clip_audio):
    mfcc = np.mean(librosa.feature.mfcc(clip_audio, sr=RATE, n_mfcc=13, n_fft=2048, hop_length=512).T,axis=0)
    X = []
    X.append(mfcc)
    X_person = scaler.transform(X)
    return X_person

def prep_data_cmd(clip_audio):
    W = []
    result = np.array([])

    mfcc = np.mean(librosa.feature.mfcc(clip_audio, sr=RATE, n_mfcc=13, n_fft=2048, hop_length=512).T, axis=0)
    result = np.hstack((result, mfcc))
    # stft = np.abs(librosa.stft(clip_audio))
    # chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=RATE).T, axis=0)
    # result = np.hstack((result, chroma))
    # mel = np.mean(librosa.feature.melspectrogram(clip_audio, sr=RATE).T, axis=0)
    # result = np.hstack((result, mel))
    W.append(result)

    X_cmd = scaler_cmd.transform(W)
    return X_cmd

def prev_result (X_person, X_cmd):
    predictions = mlp.predict(X_person)
    predictions_cmd = mlp_cmd.predict(X_cmd)
    result = grupo[int(predictions)]
    result_cmd = command[int(predictions_cmd)]
    return result, result_cmd

if __name__ == "__main__":
    # print(sd.query_devices())
    window()