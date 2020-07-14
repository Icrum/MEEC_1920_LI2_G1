import pickle
import librosa
import pyaudio
import numpy as np
import sys
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
import librosa.display
import time
import wave
import struct
import scipy.signal as sps
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from scipy.io.wavfile import write
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from sklearn.metrics import classification_report, confusion_matrix
#from pyqtgraph import PlotWidget, plot
#import pyqtgraph as pg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 2048
RECORD_SECONDS = 2.3

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


class AudioHandler(object):
    def __init__(self):
        self.FORMAT = FORMAT
        self.CHANNELS = CHANNELS
        self.RATE = RATE
        self.CHUNK = CHUNK
        self.p = None
        self.stream = None
        self.numpy_array = []
        self.numpy_arr = []
        # self.frames = []
        self.result = 8
        self.result_cmd = 8
        self.framesSize = 0

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
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

    def callback(self, in_data, frame_count, time_info, flag):
        self.numpy_array.append(in_data)
        return None, pyaudio.paContinue

    def mainloop(self):
        while (self.stream.is_active()): # if using button you can set self.stream to 0 (self.stream = 0), otherwise you can use a stop condition

            self.npSize = len(self.numpy_array)

            if self.npSize >= RATE*RECORD_SECONDS/self.CHUNK:

                self.frames = self.numpy_array[int(self.npSize-(RATE*RECORD_SECONDS/(self.CHUNK))):]

                # Save the recorded data as a WAV file
                wf = wave.open("temp.wav", 'wb')
                wf.setnchannels(1)
                wf.setsampwidth(self.p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(self.frames))
                wf.close()

                # Leitura do ficeiro para teste
                self.data, self.samplerate = sf.read("temp.wav")
                self.X_person = prep_data(self.data)
                self.X_cmd = prep_data_cmd(self.data)
                self.result, self.result_cmd = prev_result(self.X_person, self.X_cmd)
            else:
                time.sleep(0.2)

            return self.result, self.result_cmd

class MyWindow(QMainWindow):
    def __init__(self):
        super(MyWindow,self).__init__()
        self.initUI()
        self.audio = AudioHandler()


    def button_clicked(self):
        # audio = AudioHandler()
        self.audio.start()  # open the the stream
        while True:
            self.result, self.result_cmd = self.audio.mainloop()  # main operations with librosa

            print(grupo[self.result])
            print(command[self.result_cmd])
            self.user.setText(grupo[self.result])
            self.comando.setText(command[self.result_cmd])

            time.sleep(0.2)

    def button_clickedParar(self):
        self.audio.stop()

    def initUI(self):
        self.setGeometry(500, 100, 800, 600)
        self.setWindowTitle("LI Grupo 1")

        self.label = QtWidgets.QLabel(self)
        self.label.setText("Utilizador: ")
        self.label.move(150,550)
        self.user = QtWidgets.QLabel(self)
        self.user.setText("Desc")
        self.user.move(250, 550)

        self.label_2 = QtWidgets.QLabel(self)
        self.label_2.setText("Comando: ")
        self.label_2.move(400, 550)
        self.comando = QtWidgets.QLabel(self)
        self.comando.setText("Desc")
        self.comando.move(500, 550)

        self.b1 = QtWidgets.QPushButton(self)
        self.b1.setText("Iniciar")
        self.b1.clicked.connect(self.button_clicked)
        self.b1.move(10, 550)

        self.b2 = QtWidgets.QPushButton(self)
        self.b2.setText("Parar")
        self.b2.clicked.connect(self.button_clickedParar)
        self.b2.move(10, 575)

        m = PlotCanvas(self, width=5, height=2)
        m.move(0, 0)

    # def update(self):
    #     self.label.adjustSize()

class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=1, dpi=80):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.plot()


    def plot(self):
        self.data, self.samplerate = sf.read("temp.wav")
        ax = self.figure.add_subplot(111)
        ax.plot(self.data, 'r-')
        ax.set_title('Som')
        self.draw()
    
def window():
    app = QApplication(sys.argv)
    win = MyWindow()
    win.show()
    sys.exit(app.exec_())

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
    a = predictions.item(0)
    # predP = mlp.predict_proba(X_person)
    # print(mlp.predict_proba(X_person))


    predictions_cmd = mlp_cmd.predict(X_cmd)
    b = predictions_cmd.item(0)
    # result = grupo[int(predictions)]
    # result_cmd = command[int(predictions_cmd)]
    # print(mlp.predict_proba(X_cmd))
    return a, b

if __name__ == "__main__":
    print(sd.query_devices())
    window()