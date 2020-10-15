import numpy as np
from copy import deepcopy
from models.layers.ops import *
import tensorflow as tf

class EvalFunctions(object):
    """This class implements specialized operation used in the training framework"""
    def __init__(self,models):
        self.generator = models[0]
        self.discriminator = models[1]
        
    @tf.function
    def predict(self, x,training=True):
        """Returns a dict containing predictions e.g.{'predictions':predictions}"""
        if len(x)>1:
            states = x[1]
        else:
            states = None
            
        logits,states = self.model(x[0],states=states)
        return {'predictions':tf.nn.softmax(logits,axis=-1),'states':states}
    
    @tf.function
    def compute_loss(self,x,y,training=True,eval_training_steps=False):
        """Example GAN loss (untested)"""

        batches = tf.shape(x[0])[0]
        weight_decay_loss = 0
        image = x[0]
        z = tf.random.normal(shape=[batches, self.generator.latent_dim])
        image_fake = self.generator(z)
        real_logit = self.discriminator(image)
        fake_logit = self.discriminator(image_fake)

        d_loss = tf.reduce_mean(real_logit)-tf.reduce_mean(fake_logit)
        g_loss = tf.reduce_mean(fake_logit)

        total_loss = d_loss + g_loss

        if len(self.generator.losses)>0:
            weight_decay_loss += tf.add_n(self.generator.losses)
            
        if len(self.discriminator.losses)>0:
            weight_decay_loss += tf.add_n(self.discriminator.losses)
            
        total_loss += weight_decay_loss

        losses = {"d_loss":d_loss,
                "g_loss":g_loss,
                "total_loss":total_loss,
                "weight_decay_loss":weight_decay_loss}
        predictions = {"image_fake":image_fake}

        return losses,predictions

    def post_train_step(self,args):
        return
