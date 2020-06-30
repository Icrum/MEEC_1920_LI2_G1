import librosa
import numpy as np
import matplotlib.pyplot as plt

def detect_leading_silence(sound, silence_threshold=0.005):
    trim_ms = 0  # ms

    while sound[trim_ms] < silence_threshold and trim_ms < len(sound):
        trim_ms += 1

    return trim_ms

class AudioAugmentation:

    def read_audio_file(self, file_path, SAMPLE_RATE, Segundos):
        input_length = SAMPLE_RATE * Segundos
        data = librosa.core.load(file_path)[0]

        trimSignalInicio = detect_leading_silence(data)
        trimSignalFinal = detect_leading_silence(np.flip(data))

        duration = len(data)
        trimmed_sound = data[trimSignalInicio:(duration - trimSignalFinal)]



        if len(trimmed_sound) < input_length:
            sinalOut = np.pad(trimmed_sound, (0, max(0, int(input_length - len(trimmed_sound)))), "constant")

            # if int(input_length - len(trimmed_sound)) < adicaoMin:
            #     adicaoMin = int(input_length - len(trimmed_sound))
            # print("Tamanho trimmed: ", len(trimmed_sound))
            # print("Maximo ficheiro: ", max(sinalOut))
            # print("Tamanho ficheiro: ", len(sinalOut))
            # print("valor adicionado: ", adicaoMin)
        return sinalOut, int(input_length - len(trimmed_sound))

    def add_noise(self, data, a):
        # np.random.seed(a)
        noise = np.random.randn(len(data))
        data_noise = data + 0.005 * noise
        return data_noise

    def shift(self, data, a):
        return np.roll(data, 4410*a)

    def stretch(self, data, rate=1):
        input_length = 16000
        data = librosa.effects.time_stretch(data, rate)
        if len(data) > input_length:
            data = data[:input_length]
        else:
            data = np.pad(data, (0, max(0, input_length - len(data))), "constant")
        return data

