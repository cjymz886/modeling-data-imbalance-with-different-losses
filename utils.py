import tensorflow as tf
from sklearn import metrics
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.sequence import pad_sequences





def weight_cross_entropy(y_true,y_pred,category_weight):
    loss=0.0
    for label_idx,label_weight in enumerate(category_weight):
        flat_target_idx = y_true[:, label_idx]
        flat_input_idx = y_pred[:, label_idx]
        loss_idx = K.sum(K.binary_crossentropy(flat_target_idx,flat_input_idx))*label_weight
        loss += loss_idx
    return loss


def focal_loss(y_true, y_pred):
    gamma = 2.
    alpha = .2
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(K.epsilon()+pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))



def weight_focal_loss(y_true, y_pred,category_weight):
    gamma = 2.
    epsilon = 1.e-7
    alpha = tf.constant(tf.expand_dims(category_weight,-1), dtype=tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    y_t = tf.multiply(y_true, y_pred) + tf.multiply(1-y_true, 1-y_pred)
    ce = -K.log(y_t)
    weight = tf.pow(tf.subtract(1., y_t), gamma)
    fl = tf.matmul(tf.multiply(weight, ce), alpha)
    loss = tf.reduce_mean(fl)
    return loss




def multi_class_dice_loss(y_true, y_pred,logits_size):

    def _compute_dice_loss(flat_input, flat_target):
        alpha=0.2
        smooth=1e-8
        flat_input = ((1.0 - flat_input) ** alpha) * flat_input
        interection = K.sum(flat_input * flat_target, -1)
        loss = 1.0 - ((2.0 * interection + smooth) /(K.sum(flat_input) + K.sum(flat_target) + smooth))
        return loss

    loss=0.0
    for label_idx in range(logits_size):
        flat_input_idx = y_pred[:, label_idx]
        flat_target_idx = y_true[:, label_idx]
        loss_idx = _compute_dice_loss(flat_input_idx, flat_target_idx)

        loss += loss_idx

    return loss




def metric(cfg, pred_model, eval_data, word_to_id, cat_to_id,mode='train'):
    correct_num = 0
    total_num = 0

    y_test_cls=[]
    y_pred_cls=[]

    for line in eval_data:
        content = line.tokens
        label = line.label
        input_ids = [word_to_id[x] if x in word_to_id else 0 for x in content]
        inputs = pad_sequences([input_ids], value=0, padding='post', maxlen=cfg.seq_length)
        logits = pred_model.predict([inputs])
        pred_label = np.argmax(logits, axis=1)[0]

        y_test_cls.append(cat_to_id[label])
        y_pred_cls.append(pred_label)

        if pred_label == cat_to_id[label]:
            correct_num += 1
        total_num += 1
    #accuracy
    acc = correct_num / total_num

    if mode=='test':
        # evaluate
        y_test_cls = np.array(y_test_cls)
        y_pred_cls = np.array(y_pred_cls)
        print("Precision, Recall and F1-Score...")
        print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=cat_to_id))

        print("Confusion Matrix...")
        cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
        print(cm)

    return acc

