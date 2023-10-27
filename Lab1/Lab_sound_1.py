import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
import scipy.fftpack
import functions.docxSave  as ds
document = ds.DocxSave()
# Zadanie 1y
document.addHeading("Zadanie 1",0)
document.addHeading("Część 2",1)
document.addParagraph("Wczytywanie i zapis poprawnie przetestowane.")
data, fs = sf.read('Lab1/data/sound1.wav', dtype='float32')

print(data.dtype)
print(data.shape)

# sd.play(data, fs)
# status = sd.wait()

leftChannel = data[:, 0]
rightChannel = data[:, 1]
mixMono = (leftChannel + rightChannel) / 2.0


sf.write('Lab1/output/sound_L.wav', leftChannel, fs)
sf.write('Lab1/output/sound_R.wav', rightChannel, fs)
sf.write('Lab1/output/sound_mix.wav', mixMono, fs)

document.addHeading("Część 3",1)
document.addParagraph("Wyświetlenie sygnału w czasie")

fig, axs = plt.subplots(2, 1, figsize=(3, 4))

axs[0].plot(leftChannel)
axs[0].set_title('Lewy kanał')
axs[0].set_xlabel('Czas [próbki]')
axs[0].set_ylabel('Amplituda')

axs[1].plot(rightChannel)
axs[1].set_title('Prawy kanał')
axs[1].set_xlabel('Czas [próbki]')
axs[1].set_ylabel('Amplituda')

fig.tight_layout()
document.addImage(fig, width=3)

document.addParagraph("Widmo")

data, fs = sf.read('Lab1/data/sin_440Hz.wav', dtype=np.int32)

fig, axs = plt.subplots(2, 1, figsize=(3, 4))

axs[0].plot(np.arange(0,data.shape[0])/fs,data)

yf = scipy.fftpack.fft(data)
axs[1].plot(np.arange(0,fs,1.0*fs/(yf.size)),np.abs(yf))
document.addImage(fig, width=3)

document.addParagraph("Moduł widma, zmieniony rozmiar transformaty Fouriera")

fsize=2**8
fig, axs = plt.subplots(2, 1, figsize=(3, 4))

axs[0].plot(np.arange(0,data.shape[0])/fs,data)

yf = scipy.fftpack.fft(data,fsize)
axs[1].plot(np.arange(0,fs,fs/fsize),np.abs(yf))
document.addImage(fig, width=3)

document.addParagraph("Wyświetlenie modułu widma, wzięcie tylko połowy")

fig, axs = plt.subplots(2, 1, figsize=(3, 4))

axs[0].plot(np.arange(0,data.shape[0])/fs,data)
yf = scipy.fftpack.fft(data,fsize)
axs[1].plot(np.arange(0,fs/2,fs/fsize),np.abs(yf[:fsize//2]))
document.addImage(fig, width=3)


document.addParagraph("Przeskalowanie wartości widma do skali decybelowej (dB)")

fig, axs = plt.subplots(2, 1, figsize=(3, 4))


axs[0].plot(np.arange(0,data.shape[0])/fs,data)
yf = scipy.fftpack.fft(data,fsize)
axs[1].plot(np.arange(0,fs/2,fs/fsize),20*np.log10( np.abs(yf[:fsize//2])))

document.addImage(fig, width=3)

document.addHeading("Zadanie 2",0)

def plotAudio(Signal, Fs, TimeMargin=[0, 0.02]):
    start = int(TimeMargin[0] * Fs)
    end = int(TimeMargin[1] * Fs)
    signalFragment = Signal[start:end]
    # time in sec
    time = np.arange(0, len(signalFragment) / Fs, 1 / Fs)
    fig, axs = plt.subplots(2, 1, figsize=(3, 4))
    axs[0].plot(time, signalFragment)
    axs[0].set_title('Sygnał dźwiękowy')
    axs[0].set_xlabel('Czas [s]')
    axs[0].set_ylabel('Amplituda')
    # widmo w dB
    spectrum = 20 * np.log10(np.abs(np.fft.fft(signalFragment)) / len(signalFragment))
    freq = np.fft.fftfreq(len(signalFragment), 1 / Fs)
    # połowa widma
    halfSpectrum = spectrum[:len(spectrum)//2]
    halfFreq = freq[:len(freq)//2]
    axs[1].plot(halfFreq, halfSpectrum)
    axs[1].set_title('Widmo dźwiękowe (połowa)')
    axs[1].set_xlabel('Częstotliwość [Hz]')
    axs[1].set_ylabel('Amplituda [dB]')
    fig.tight_layout()
    return fig

Signal, Fs = sf.read('Lab1/data/sin_440Hz.wav')
fig = plotAudio(Signal, Fs)

document.addImage(fig, width=3)
document.addHeading("Zadanie 3",0)
files=["sin_60Hz.wav",
"sin_440Hz.wav",
"sin_8000Hz.wav",
"sin_combined.wav"]
fsizeValues = [2**8, 2**12, 2**16]
for filename in files:
    for fsize in fsizeValues:
            data, fs = sf.read("Lab1/data/"+filename, dtype=np.int32)
            # obliczanie widma dla danego fsize
            yf = np.fft.fft(data, fsize)
            freqs = np.fft.fftfreq(fsize, 1.0 / fs)
            
            maxAmplitudeIndex = np.argmax(np.abs(yf))
            maxAmplitude = np.abs(yf[maxAmplitudeIndex])
            maxFrequency = freqs[maxAmplitudeIndex]
            
            fig = plotAudio(Signal, Fs)
            document.addImage(fig, width=3)
            document.addHeading(f'Analiza dla fsize={fsize}', level=1)
            document.addParagraph(f'Najwyższa amplituda: {maxAmplitude}')
            document.addParagraph(f'Dla częstotliwości: {maxFrequency} Hz')

document.save("LabSound1.docx")
