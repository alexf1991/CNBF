import tensorflow as tf

class EvalFunctions(object):
    """This class implements specialized operation used in the training framework"""
    def __init__(self,models):
        self.classifier = models[0]

    @tf.function
    def predict(self, x,training=True):
        """Returns a dict containing predictions e.g.{'predictions':predictions}"""
        logits_classifier = self.classifier(x[0], training=training)
        return {'predictions':tf.nn.softmax(logits_classifier,axis=-1)}

    def accuracy(self,pred,y):
        correct_predictions = tf.cast(tf.equal(tf.argmax(pred,axis=-1), 
                                        tf.argmax(y[0],axis=-1)),tf.float32)
        return tf.reduce_mean(correct_predictions)
        
    @tf.function
    def compute_loss(self, x, y, training=True):
        """Has to at least return a dict containing the total loss and a prediction dict e.g.{'total_loss':total_loss},{'predictions':predictions}"""
        logits_classifier = self.classifier(x[0], training=training)
        predictions = tf.nn.softmax(logits_classifier,axis=-1)
        # Cross entropy losses
        class_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                y[0],
                logits_classifier,
                axis=-1,
            ))

        if len(self.classifier.losses) > 0:
            weight_decay_loss = tf.add_n(self.classifier.losses)
        else:
            weight_decay_loss = 0.0
        total_loss = class_loss
        total_loss += weight_decay_loss
        
        accuracy = self.accuracy(logits_classifier,y)
        
        scalars = {'class_loss':class_loss,'weight_decay_loss':weight_decay_loss,'total_loss':total_loss,"accuracy":accuracy}
        
        predictions = {'predictions':predictions,"logits":logits_classifier}
        
        return scalars, predictions
    
    def post_train_step(self,args):
        return
