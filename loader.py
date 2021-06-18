import codecs
import numpy as np
import re
from collections import  Counter
from collections import namedtuple
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight


def read_file(file_dir):
    """
    读入数据文件，将每条数据的文本和label存入各自列表中
    """
    re_han = re.compile(u"([\u4E00-\u9FD5a-zA-Z]+)")  # 去掉标点符号和数字类型的字符

    SentInst = namedtuple('SentInst', 'tokens label')
    data =[]
    with codecs.open(file_dir ,'r' ,encoding='utf-8') as f:
        for line in f:
            label ,text =line.split('\t')
            content =[]
            for w in text[:400]:
                if re_han.match(w):
                    content.append(w)
            sent_inst = SentInst(content, label)
            data.append(sent_inst)
    return data


def build_vocab(file_dirs,vocab_dir,vocab_size=6000):
    """
    利用训练集和测试集的数据生成字级的词表
    """
    all_data = []
    for filename in file_dirs:
        for line in read_file(filename):
            content= line.tokens
            all_data.extend(content)
    counter=Counter(all_data)
    count_pairs=counter.most_common(vocab_size-1)
    words,_=list(zip(*count_pairs))
    words=['<PAD>']+list(words)

    with codecs.open(vocab_dir,'w',encoding='utf-8') as f:
        f.write('\n'.join(words)+'\n')



def read_category():
    categories = ['Art', 'Literature', 'Education', 'Philosophy', 'History', 'Space', 'Energy', 'Electronics',
                  'Communication', 'Computer','Mine','Transport','Enviornment','Agriculture','Economy',
                  'Law','Medical','Military','Politics','Sports']
    cat_to_id=dict(zip(categories,range(len(categories))))
    return cat_to_id


def read_vocab(vocab_dir):
    words=codecs.open(vocab_dir,'r',encoding='utf-8').read().strip().split('\n')
    word_to_id=dict(zip(words,range(len(words))))
    return word_to_id



def compute_weight(data,cat_to_id):

    label_set=[]
    for line in data:
        label=line.label
        label_set.append(label)

    class_weight='balanced'
    category=np.array(list(cat_to_id.keys()))
    weight = compute_class_weight(class_weight=class_weight,classes=category, y=label_set)
    print('category_weight',weight)

    return weight


def count_data(data,cat_to_id):

    count_dict=dict(zip(cat_to_id.keys(),[0]*len(cat_to_id)))

    for line in data:
        label=line.label
        count_dict[label]+=1

    print('total num',len(data))
    print(count_dict)





class data_generator:
    def __init__(self, cfg, data, word_to_id, cat_to_id):
        self.cfg = cfg
        self.data = data
        self.steps = len(self.data) // self.cfg.batch_size
        if len(self.data) % self.cfg.batch_size != 0:
            self.steps += 1
        self.word_to_id = word_to_id
        self.cat_to_id = cat_to_id

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            text_batch, label_batch = [], []
            for idx in idxs:
                line = self.data[idx]
                content = line.tokens
                label = line.label

                input_ids = [self.word_to_id[x] if x in self.word_to_id else 0 for x in content]

                label_ids = self.cat_to_id[label]

                text_batch.append(input_ids)
                label_batch.append(label_ids)

                if len(text_batch) == self.cfg.batch_size or idx == idxs[-1]:
                    text_batch = pad_sequences(text_batch, value=0, padding='post', maxlen=self.cfg.seq_length)
                    label_batch = to_categorical(label_batch, num_classes=self.cfg.num_classes)

                    yield [text_batch, label_batch], None
                    text_batch, label_batch = [], []