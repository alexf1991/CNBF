# -*- coding: utf-8 -*-
__author__ = "Lukas Pfeifenberger"


import time
import glob
import argparse
import json
import os
import sys
import copy
from scipy import signal
import tensorflow as tf
import numpy as np
import pyroomacoustics as pra
import asyncio
sys.path.append(os.path.abspath('../'))

from loaders.audio_loader import audio_loader
from loaders.rir_generator import rir_generator
from algorithms.audio_processing import *
from utils.mat_helpers import *


def blackman_window_fn(wlen, dtype):
    analysis_window = signal.blackman(wlen)
    return tf.cast(analysis_window, dtype)
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
class feature_generator(object):


    #--------------------------------------------------------------------------
    def __init__(self, config, set='train',steps=2000,buffer_size=20):

        self.set = set
        self.config = config
        self.fs = config['fs']
        self.wlen = config['wlen']
        self.shift = config['shift']
        self.samples = int(self.fs*config['duration'])
        self.nfram = int(np.ceil( (self.samples-self.wlen+self.shift)/self.shift ))
        self.nbin = int(self.wlen/2+1)
        self.steps = steps
        self.buffer_size = buffer_size

        self.nsrc = config['nsrc']
        assert(self.nsrc == 2)                              # only 2 sources are supported

        self.audio_loader = audio_loader(config, set)
        self.rgen = rir_generator(config, set)
        self.nmic = self.rgen.nmic
        self.ffts = {}

    def generate_fast(self):
        # Not yet working
        for i in range(self.steps//self.buffer_size):
            a,b,c,d = self.generate_multi_mixtures()
            for i in range(self.buffer_size):
                yield a[i],b[i],c[i],d[i]

    def generate(self):
        for i in range(self.steps):
            yield self.generate_mixture()

    def rfft(self,x,n,transpose=False):
        if transpose:
            x = tf.transpose(x,[1,0])
        x = tf.signal.rfft(x,fft_length=[n])
        if transpose:
            x = tf.transpose(x,[1,0])
        return x

    def irfft(self,x,n,transpose=False):
        if transpose:
            x = tf.transpose(x,[1,0])
        x = tf.signal.irfft(x,fft_length=[n])
        #if transpose:
        #    x = tf.transpose(x,[1,0])
        return x



    def mstft(self,x,wlen=1024, shift=256, window=blackman_window_fn,transpose=True,is_2D=False):
        #if transpose:
        #    x = tf.transpose(x,[1,0])

        samples = x.shape[-1]
        nfram = int(np.ceil((samples - wlen + shift) / shift))
        samples_padding = nfram * shift + wlen - shift - samples
        if is_2D:
            x = tf.pad(x,[[0,0],[0,samples_padding]])
        else:
            x = tf.pad(x,[[0,samples_padding]])
        x = tf.signal.stft(x,wlen,shift,window_fn = window)
        #if transpose:
        #    x = tf.transpose(x,[1,2,0])

        return x

    #---------------------------------------------------------
    def generate_multi_mixtures(self,):
        #Not yet working
        refs = []
        for i in range(self.buffer_size):
            hs, hn = self.rgen.load_rirs()
            s = self.audio_loader.concatenate_random_files()                    # shape = (samples,)
            n = self.audio_loader.concatenate_random_files()                    # shape = (samples,)
            ref = copy.deepcopy(s)
            if i == 0:
                hsn_tot = tf.concat([hs,hn],axis=-1)
                sn_tot = tf.stack([s, n], axis=-1)
            else:
                hsn_tot = tf.concat([hsn_tot,hs, hn], axis=-1)
                sn_tot = tf.concat([sn_tot,s[...,tf.newaxis], n[...,tf.newaxis]], axis=-1)

            refs.append(ref)

        refs = tf.stack(refs,axis=0)

        Fhsn_tot = self.rfft(hsn_tot, n=self.samples,transpose=True)
        Fsn_tot = self.rfft(sn_tot, n=self.samples,transpose=True)

        Fsn_tot = tf.tile(tf.expand_dims(Fsn_tot,-1),[1,1,self.nmic])
        Fsn_tot = tf.reshape(Fsn_tot,Fhsn_tot.shape)

        Fs = Fhsn_tot*Fsn_tot

        sn_tot = self.irfft(Fsn_tot, n=self.samples,transpose=True)

        Fsn_tot = self.mstft(sn_tot,self.wlen,self.shift,is_2D=True)
        Fsn_tot = tf.reshape(Fsn_tot,[self.buffer_size,2*self.nmic,self.nfram,self.nbin])
        Fsn_tot = tf.transpose(Fsn_tot,perm=[0,2,3,1])
        Frefs = self.mstft(refs,self.wlen,self.shift,is_2D=True)                            # shape = (nbin, nfram)
        Frefs = tf.reshape(Frefs, [self.buffer_size,self.nfram, self.nbin])

        Fs = self.rgen.whiten_data(Fsn_tot[:,:,:,:self.nmic])
        Fn = self.rgen.whiten_data(Fsn_tot[:,:,:,self.nmic:])

        return Fs, Fn,refs,Frefs

    #---------------------------------------------------------
    def generate_mixture(self,):

        hs, hn = self.rgen.load_rirs()
        s = self.audio_loader.concatenate_random_files()                    # shape = (samples,)
        n = self.audio_loader.concatenate_random_files()                    # shape = (samples,)
        ref = copy.deepcopy(s)
        hsn_tot = tf.concat([hs,hn],axis=-1)
        Fhsn_tot = self.rfft(hsn_tot, n=self.samples,transpose=True)
        Fhs = Fhsn_tot[...,:self.nmic]
        Fhn = Fhsn_tot[...,self.nmic:]
        #Fhs = self.rfft(hs, n=self.samples,transpose=True)                              # shape = (samples/2+1, nmic)
        #Fhn = self.rfft(hn, n=self.samples,transpose=True)                              # shape = (samples/2+1, nmic)

        sn_tot = tf.stack([s,n],axis=-1)
        Fsn_tot = self.rfft(sn_tot, n=self.samples,transpose=True)
        Fs = Fsn_tot[...,0]
        Fn = Fsn_tot[...,1]
        #Fs = self.rfft(s, n=self.samples,transpose=False)                                # shape = (samples/2+1,)
        #Fn = self.rfft(n, n=self.samples,transpose=False)                                # shape = (samples/2+1,)

        Fs = Fhs*Fs[:,tf.newaxis]
        Fn = Fhn*Fn[:,tf.newaxis]


        Fsn_tot = tf.concat([Fs,Fn],axis=-1)
        sn_tot = self.irfft(Fsn_tot, n=self.samples,transpose=True)
        Fsn_tot = self.mstft(sn_tot,self.wlen,self.shift)
        Fsn_tot = tf.transpose(Fsn_tot, [1, 2, 0])
        Fs = Fsn_tot[...,:self.nmic]
        Fn = Fsn_tot[...,self.nmic:]
        #s = self.irfft(Fs, n=self.samples, transpose=True)                               # shape = (samples, nmic)
        #n = self.irfft(Fn, n=self.samples, transpose=True)                               # shape = (samples, nmic)
        #Fs = self.mstft(s, self.wlen, self.shift)                              # shape = (nmic, nfram, nbin)
        #Fn = self.mstft(n, self.wlen, self.shift)                              # shape = (nmic, nfram, nbin)

        Fref = self.mstft(ref,self.wlen,self.shift,is_2D=False)                            # shape = (nbin, nfram)
        Fref = tf.transpose(Fref, [1, 2, 0])
        Fs = self.rgen.whiten_data(Fs)
        Fn = self.rgen.whiten_data(Fn)

        return Fs, Fn,ref,Fref



    #---------------------------------------------------------
    def generate_mixtures(self, nbatch=10):

        Fs = np.zeros(shape=(nbatch, self.nfram, self.nbin, self.nmic), dtype=np.complex64)
        Fn = np.zeros(shape=(nbatch, self.nfram, self.nbin, self.nmic), dtype=np.complex64)
        for b in np.arange(nbatch):

            Fs[b,...], Fn[b,...] = self.generate_mixture()

        return Fs, Fn





#---------------------------------------------------------
#---------------------------------------------------------
if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='mcss feature generator')
    parser.add_argument('--config_file', help='name of json configuration file', default='../cnbf.json')
    args = parser.parse_args()


    with open(args.config_file, 'r') as f:
        config = json.load(f)


    fgen = feature_generator(config, set='train')


    t0 = time.time()
    Fs, Fn = fgen.generate_mixture()
    t1 = time.time()
    print(t1-t0)

    data = {
            'Fs': Fs,
            'Fn': Fn,
           }
    save_numpy_to_mat('../matlab/fgen_check.mat', data)



