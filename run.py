import pickle
import os
import sys
import numpy as np
from loader import *
from model import *



os.environ["CUDA_VISIBLE_DEVICES"] = "-1"



def train():
    data = read_file(cfg.train_dir)

    # count_data(data,cat_to_id)

    if cfg.use_weight:
        category_weight= compute_weight(data,cat_to_id)
        cfg.category_weight=category_weight


    random_order = list(range(len(data)))
    np.random.seed(cfg.random_seed)
    np.random.shuffle(random_order)
    data = [data[i] for i in random_order]

    train_data = data[:-1000]
    val_data = data[-1000:]

    steps = len(train_data) // cfg.batch_size
    data_manager = data_generator(cfg, train_data, word_to_id, cat_to_id)

    evaluator = Evaluate(cfg, pred_model, val_data, word_to_id, cat_to_id)
    history = final_model.fit(data_manager.__iter__(),
                                        steps_per_epoch=steps,
                                        epochs=cfg.epochs,
                                        callbacks=[evaluator],
                                        verbose=2
                                        )

    with open(cfg.history_dir, 'wb') as f:
        pickle.dump(history.history, f)


def test():
    test_data=read_file(cfg.test_dir)
    final_model.load_weights(cfg.save_dir)
    accuracy= metric(cfg,pred_model,test_data,word_to_id, cat_to_id,mode='test')
    print(f'{accuracy}')



if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ['train', 'test']:
        raise ValueError("""usage: python run_rnn.py [train / test]""")

    cfg = TextConfig()
    pred_model,final_model = bulid_model(cfg)

    file_dirs = [cfg.train_dir, cfg.test_dir]
    if not os.path.exists(cfg.vocab_dir):
        build_vocab(file_dirs, cfg.vocab_dir, cfg.vocab_size)

    word_to_id =read_vocab(cfg.vocab_dir)
    cat_to_id =read_category()

    if sys.argv[1] == 'train':
        train()
    else:
        test()