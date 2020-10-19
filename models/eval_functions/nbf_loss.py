import tensorflow as tf
import os
from utils.mat_helpers import *
from models.ops.complex_ops import *
from models.layers.complex_layers import *
from models.layers.kernelized_layers import *
from algorithms.audio_processing import *
from utils.matplotlib_helpers import *


class EvalFunctions(object):
    """This class implements specialized operation used in the training framework"""
    def __init__(self,models):
        self.nbf = models[0]
        self.name = "test_sample"

    #@tf.function
    def predict(self, x,training=True):
        """Returns a dict containing predictions e.g.{'predictions':predictions}"""
        Fs = x[0]
        Fn = x[1]
        Fys, Fyn, cost,_,_ = self.nbf(Fs, Fn,training=False)
        Fys = Fys.numpy()
        Fyn = Fyn.numpy()
        Fs = Fs.numpy()
        Fn = Fn.numpy()
        Fs = Fs[0, ..., 0].T  # input (desired source)
        Fn = Fn[0, ..., 0].T  # input (unwanted source)
        Fys = Fys[0, ...].T  # output (desired source)
        Fyn = Fyn[0, ...].T  # output (unwanted source)
        Fz = Fs + Fn  # noisy mixture
        Fy = Fys + Fyn  # enhanced output

        data = (Fz, Fy,Fs)
        filenames = (
            os.path.join(self.nbf.config['predictions_path'] , self.name + '_noisy.wav'),
            os.path.join(self.nbf.config['predictions_path'] , self.name + '_enhanced.wav'),
            os.path.join(self.nbf.config['predictions_path'], self.name + '_clean.wav')
        )
        pesq_score = convert_and_save_wavs(data, filenames)

        Lz = (20 * np.log10(np.abs(Fs) + 1e-1) - 20 * np.log10(np.abs(Fn) + 1e-1)) / 30
        Ly = (20 * np.log10(np.abs(Fys) + 1e-1) - 20 * np.log10(np.abs(Fyn) + 1e-1)) / 30
        legend = ('noisy', 'enhanced')
        clim = (-1, +1)
        filename = os.path.join(self.nbf.config['predictions_path'] , self.name + '_prediction.png')
        draw_subpcolor((Lz, Ly), legend, clim, filename)
        return {"predictions":[Lz,Ly],"pesq_score":pesq_score}

    def save_prediction(self,x):

        Fs = x[0]
        Fn = x[1]
        Fys, Fyn, cost,_,_ = self.nbf(Fs, Fn,training=False)
        Fys = Fys.numpy()
        Fyn = Fyn.numpy()
        Fs = Fs.numpy()
        Fn = Fn.numpy()

        data = {
            'Fs': np.transpose(Fs, [0, 2, 1, 3])[0, :, :, 0],  # shape = (nbin, nfram)
            'Fn': np.transpose(Fn, [0, 2, 1, 3])[0, :, :, 0],  # shape = (nbin, nfram)
            'Fys': np.transpose(Fys, [0, 2, 1])[0, :, :],  # shape = (nbin, nfram)
            'Fyn': np.transpose(Fyn, [0, 2, 1])[0, :, :],  # shape = (nbin, nfram)
        }
        save_numpy_to_mat(self.predictions_file, data)

    def accuracy(self,pred,y):
        correct_predictions = tf.cast(tf.equal(tf.argmax(pred,axis=-1), 
                                        tf.argmax(y[0],axis=-1)),tf.float32)
        return tf.reduce_mean(correct_predictions)
        
    @tf.function
    def compute_loss(self, x, y, training=True):
        """Has to at least return a dict containing the total loss and a prediction dict e.g.{'total_loss':total_loss},{'predictions':predictions}"""
        Fys, Fyn, cost,opt_loss,ws_loss = self.nbf(x[0],x[1], training=training)

        if len(self.nbf.losses) > 0:
            weight_decay_loss = tf.add_n(self.nbf.losses)
        else:
            weight_decay_loss = 0.0

        total_loss = opt_loss
        total_loss += weight_decay_loss
        
        scalars = {'bf_loss':cost,'opt_loss':opt_loss,'ws_loss':ws_loss,'weight_decay_loss':weight_decay_loss,
                   'total_loss':total_loss}
        
        predictions = {'Fys':Fys,"Fyn":Fyn}
        
        return scalars, predictions
    
    def post_train_step(self,args):
        return
