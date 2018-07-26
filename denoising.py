# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 19:53:19 2018

@author: matth
"""

from scipy.io import wavfile
from scipy.signal import resample

#import StringIO
import base64
import struct
  
# change to the shogun-data directoy
import os
os.chdir('C:/Users/matth/Desktop/sound')

#%pylab inline
import pylab as pl
import numpy as np

from IPython.core.display import HTML
from shogun.Features  import RealFeatures
from shogun.Converter import Jade




def load_wav(filename,samplerate=44100):

    # loading the file
    rate, data = wavfile.read(filename)

    # conversion of the stereo audio into a mono audio
    if len(data.shape) > 1:
        data = data[:,0]/2 + data[:,1]/2

    # re-interpose samplerate    
    ratio = float(samplerate) / float(rate)
    data = resample(data, len(data) * ratio)

    return samplerate, data.astype(np.int16)

def wavPlayer(data, rate):
    
    buffer = six.moves.StringIO()
    buffer.write(b'RIFF')
    buffer.write(b'\x00\x00\x00\x00')
    buffer.write(b'WAVE')

    buffer.write(b'fmt ')
    if data.ndim == 1:
        noc = 1
    else:
        noc = data.shape[1]
    bits = data.dtype.itemsize * 8
    sbytes = rate*(bits // 8)*noc
    ba = noc * (bits // 8)
    buffer.write(struct.pack('<ihHIIHH', 16, 1, noc, rate, sbytes, ba, bits))

    # data block
    buffer.write(b'data')
    buffer.write(struct.pack('<i', data.nbytes))

    if data.dtype.byteorder == '>' or (data.dtype.byteorder == '=' and sys.byteorder == 'big'):
        data = data.byteswap()

    buffer.write(data.tostring())
    # return buffer.getvalue()
    # Determine file size and place it in correct
    # position at start of the file.
    size = buffer.tell()
    buffer.seek(4)
    buffer.write(struct.pack('<i', size-8))

    val = buffer.getvalue()

    src = """
    <head>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
    <title>Simple Test</title>
    </head>

    <body>
    <audio controls="controls" style="width:600px" >
      <source controls src="data:audio/wav;base64,{base64}" type="audio/wav" />
      Your browser does not support the audio element.
    </audio>
    </body>
    """.format(base64=base64.encodestring(val))
    display(HTML(src))


# loading the first file
fs1,s1 = load_wav('ak471.wav')
# plot the audio file
pl.figure(figsize=(7,2))
pl.plot(s1)
pl.title('Signal 1')
pl.show()
# play the audio file
wavPlayer(s1, fs1)  

# loading the second file
fs2,s2 = load_wav('siren.wav') 
# plot the audio file
pl.figure(figsize=(6.75,2))
pl.plot(s2)
pl.title('Signal 2')
pl.show()
# play the audio file
wavPlayer(s2, fs2)


# Adjust for different clip lengths
fs = fs1
length = max([len(s1), len(s2)])

s1.resize((length,1), refcheck=False)
s2.resize((length,1), refcheck=False)

S = (np.c_[s1, s2]).T      #transposition for having an input source matrix with the right size in accord to the chosen mixing matrix'

# Mixing Matrix
A = np.array([[1, 0.5],
              [0.5, 1]])
print ('Mixing Matrix:')
print (A.round(2))

# Mixed Signals
X = np.dot(A,S)

# Exploring Mixed Signals
for i in range(X.shape[0]):
    pl.figure(figsize=(6.75,2))
    pl.plot((X[i]).astype(np.int16))
    pl.title('Mixed Signal %d' % (i+1))
    pl.show()
    wavPlayer((X[i]).astype(np.int16), fs)
    
# Convert to features for shogun
mixed_signals = RealFeatures((X).astype(np.float64))

# Separating with JADE
jade = Jade()
signals = jade.apply(mixed_signals)

S_ = signals.get_feature_matrix()

A_ = jade.get_mixing_matrix()
A_ = A_ / A_.sum(axis=0)

print ('Estimated Mixing Matrix:')
print (A_)

# Show separation results

# Separated Signal i
gain = 4000
for i in range(S_.shape[0]):
    pl.figure(figsize=(6.75,2))
    pl.plot((gain*S_[i]).astype(np.int16))
    pl.title('Separated Signal %d' % (i+1))
    pl.show()
    wavPlayer((gain*S_[i]).astype(np.int16), fs)