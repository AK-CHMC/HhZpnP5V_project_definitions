import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import (
    TruePositives as TP,
    FalsePositives as FP,
    TrueNegatives as TN,
    FalseNegatives as FN,
    BinaryAccuracy,
    Precision,
    Recall,
    AUC,
    Metric
)


# original function for matthew's correlation coefficient 
# See note with MCC class
def mcc(y_true, y_pred):
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))

    num = tp * tn - fp * fn
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return num / K.sqrt(den + K.epsilon())



def get_metric_list():
    """
    get_metric_list will return a list of
    keras metrics that can then be passed to the
    '.compile' function of a keras model
    """
    METRICS = [TP(name='tp'),
               FP(name='fp'),
               TN(name='tn'),
               FN(name='fn'), 
               BinaryAccuracy(name='accuracy'),
               Precision(name='precision'),
               Recall(name='recall'),
               AUC(name='auc'),
               AUC(name='prauc', curve='PR'),
               mcc]
    return METRICS


"""
remake of mcc function so that it would properly calculate the mcc value
across each epoch. 

Had the realization that the way keras aggregates custom metrics defined by
a function is to store the outputs from each batch in an epoch and then use 
a aggregation operator at the end to combine them. 
The default one chosen by keras is the mean.

Since I don't believe mcc values can be combined by simply averaging them together,
I created a custom metric class. This way I could be in charge of the way that
the mcc value is updated per each batch over the course of the evaluation
"""
class MCC(Metric):
    def __init__(self, name='mcc', **kwargs):
        # Initialize the MCC class as a subclass of Metric, and set the name
        super(MCC, self).__init__(name=name, **kwargs)
        # Initialize the weights for the running counts of each metric
        self.running_tp = self.add_weight(name='tp', initializer='zeros')
        self.running_tn = self.add_weight(name='tn', initializer='zeros')
        self.running_fp = self.add_weight(name='fp', initializer='zeros')
        self.running_fn = self.add_weight(name='fn', initializer='zeros')
      
    # reset_state is called at the beginning of each epoch
    def reset_state(self):
        # Reset the running counts of each metric to zero
        self.running_tp.assign(0)
        self.running_tn.assign(0)
        self.running_fp.assign(0)
        self.running_fn.assign(0)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Cast the true and predicted values to float32
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Wrapper function to clip x values between 0 and 1, round them, and sum them up
        func = lambda x: K.sum(K.round(K.clip(x, 0, 1)))
        # Calculate true positives, true negatives, false positives, and false negatives
        tp = func(y_true * y_pred)
        tn = func((1 - y_true) * (1 - y_pred))
        fp = func((1 - y_true) * y_pred)
        fn = func(y_true * (1 - y_pred))

        # Add the counts of true positives, true negatives, false positives, and false negatives to the running weights
        self.running_tp.assign_add(tp)
        self.running_tn.assign_add(tn)
        self.running_fp.assign_add(fp)
        self.running_fn.assign_add(fn)

    def result(self):
        # Get the values for true positives, true negatives, false positives, and false negatives from the running weights
        tp = self.running_tp
        tn = self.running_tn
        fp = self.running_fp
        fn = self.running_fn

        # Calculate numerator and denominator for the mcc
        num = (tp * tn) - (fp * fn)
        den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    
        # Return the MCC, taking care to add a small value (epsilon) to the denominator to avoid division by zero
        return num / (K.sqrt(den) + K.epsilon())


