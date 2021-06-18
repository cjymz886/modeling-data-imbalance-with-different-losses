import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback
import numpy as np
from  utils import *




class TextConfig():
    embedding_size = 100
    vocab_size = 6000

    seq_length = 300
    num_classes = 20

    num_filters = 128
    filter_sizes = [2, 3, 4]
    random_seed = 2009

    keep_prob = 0.5
    lr = 1e-3

    batch_size = 32
    epochs = 30

    loss_type='normal'  #normal,focal_loss,dice_loss
    use_weight=False
    category_weight=np.array([1.0]*num_classes,dtype=np.float32)  #init category weight

    train_dir = './data/train.txt'
    test_dir = './data/test.txt'
    vocab_dir = './data/vocab.txt'

    save_dir='./save_models/'+loss_type+'/best_model.weights'
    history_dir='./save_models/'+loss_type+'/history.pickle'




def bulid_model(cfg):
    text_in = layers.Input(shape=(None,))
    label_in = layers.Input(shape=(20))

    embeddings = layers.Embedding(cfg.vocab_size,
                                  cfg.embedding_size,
                                  embeddings_initializer='uniform',
                                  trainable=True,
                                  input_length=cfg.seq_length)

    inputs = embeddings(text_in)
    inputs = tf.expand_dims(inputs, -1)
    # conv
    cnns = []
    for size in cfg.filter_sizes:
        conv = layers.Conv2D(filters=cfg.num_filters, kernel_size=(size, cfg.embedding_size),
                             strides=1, padding='valid', activation='relu',
                             kernel_regularizer=tf.keras.regularizers.l2(0.001))(inputs)
        pool = layers.MaxPool2D(pool_size=(cfg.seq_length - size + 1, 1), padding='valid')(conv)
        cnns.append(pool)

    text_features = layers.concatenate(cnns, axis=-1)
    text_features = tf.reshape(text_features, (-1, len(cfg.filter_sizes) * cfg.num_filters))
    text_features = layers.Dropout(cfg.keep_prob)(text_features)

    # pred
    pred = layers.Dense(cfg.num_classes, activation='softmax')(text_features)

    # model
    pred_model = tf.keras.models.Model([text_in], [pred])
    final_model = tf.keras.models.Model([text_in, label_in], [pred])

    # loss
    if cfg.loss_type=='focal_loss':
        if cfg.use_weight:
            loss=weight_focal_loss(label_in,pred,cfg.category_weight)
        else:
            loss=focal_loss(label_in,pred)
    elif cfg.loss_type=='dice_loss':
        loss=multi_class_dice_loss(label_in,pred,cfg.num_classes)
    else:
        if cfg.use_weight:
            loss= weight_cross_entropy(label_in,pred,cfg.category_weight)
        else:
            loss = K.sum(tf.keras.losses.categorical_crossentropy(label_in, pred))

    final_model.add_loss(loss)
    final_model.compile(optimizer=Adam(cfg.lr))
    #     model.summary()
    return pred_model, final_model


class Evaluate(Callback):
    def __init__(self, cfg, pred_model, eval_data, word_to_id, cat_to_id, min_delta=1e-4, patience=7):
        self.patience = patience
        self.min_delta = min_delta
        self.cfg = cfg
        self.monitor_op = np.greater
        self.pred_model = pred_model
        self.eval_data = eval_data
        self.word_to_id = word_to_id
        self.cat_to_id = cat_to_id


    def on_train_begin(self, logs=None):
        self.step = 0
        self.wait = 0
        self.stopped_epoch = 0
        self.warmup_epochs = 2
        self.best = -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        acc = metric(self.cfg, self.pred_model, self.eval_data, self.word_to_id, self.cat_to_id)
        if self.monitor_op(acc - self.min_delta, self.best) or self.monitor_op(self.min_delta, acc):
            self.best = acc
            self.wait = 0
            self.model.save_weights(self.cfg.save_dir)
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
            self.cfg.lr = self.cfg.lr * 0.9
        print('\nacc: %.4f,  best acc: %.4f' % (acc, self.best))

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))


