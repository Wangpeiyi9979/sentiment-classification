# -*- coding: utf-8 -*-

class Config():

    glove_param = {'use_id': False,
                   'requires_grad':True,
                   'vocab_size':18766,
                   'glove_dim':300,
                   'word2id_file':'./data/glove/word2id.npy',   # path/None
                   'glove_file':'./data/glove/glove_300d.npy'}

    elmo_param = {'elmo_dim': 512,
                  'requires_grad':False,
                  'elmo_options_file':'./data/elmo/elmo_2x2048_256_2048cnn_1xhighway_options.json',
                  'elmo_weight_file':'./data/elmo/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5'}

    max_len = 50
    model = 'TextCNN'
    enc_method = 'rnn'     # cnn/rnn/gru/lstm
    emb_method = 'glove'  # elmo/glove/elmo_glove
    att_method = 'Hdot'    # Hdot/Tdot1/Tdot2/Cat
    use_mask = True

    filters_num = 100
    filters = [3, 4, 5]
    num_labels = 2

    hidden_size = 100
    bidirectional = True
    q_num = 1
    q_dim = 100

    seed = 200
    use_gpu = True
    gpu_id = 0
    dropout = 0.5
    epochs = 10
    test_size = 0.1
    lr = 1e-3
    weight_decay = 1e-4
    batch_size = 64


def parse(self, kwargs):
        '''
        user can update the default hyperparamter
        '''
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise Exception('opt has No key: {}'.format(k))
            setattr(self, k, v)

        print('*************************************************')
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print("{} => {}".format(k, getattr(self, k)))

        print('*************************************************')


Config.parse = parse
opt = Config()
