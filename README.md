## 项目目录

```
├── config.py             # 参数
├── data              # 数据目录
│ ├── elmo       # elmo所需数据
│ ├──├── elmo_2x2048_256_2048cnn_1xhighway_options.json
│ ├──├── elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5
│ ├── glove              # glove所需数据
│ ├──├── glove_300d.npy
│ ├──├── word2id.npy
│ ├── rt-polarity.neg     # 负面评价原数据
│ ├── rt-polarity.pos   # 正面评价原数据
├── main.py       # 主文件
├── data_pro.py       # 数据处理文件
├── models               # 模型目录
│ ├── BasicModule.py
│ ├── __init__.py
│ ├── model.py
├── modules 
│ ├── __init__.py    
│ ├── embedder.py     # 封装了embedding层
│ ├── attender.py     # 封装了attention层
│ ├── encoder.py     # 封装了encoder层
├── README.md
```
----
代码框架参考陈云的指南。数据模型分开，参数/配置单独文件，使用`fire`管理命令行参数，利于参数修改.

-----

## 环境
- Ubuntu 16.04
- Python 3.X
- Pytorch 1.X
- fire

## 如何运行这个例子
- 在[百度网盘](https://pan.baidu.com/s/1fPM-picx20bpAGzkXWdDjQ) 中下载glove词向量`glove_300d.npy`，elmo配置文件`elmo_2x2048_256_2048cnn_1xhighway_options.json`和`elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5`, 单词id映射文件`word2id.npy`。放入上方所示文件目录。
- 密码:`6av4`

- 回到主目录, 执行：

```
python main.py train
```

----

## 各封装层使用说明


-------

### embedding 层操作说明.
#### 需求
在`config.py`中需要有两个字典

```
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
```

- 参数说明:
    - glove_param
        - use_id: `bool`，True表示使用Tensor代表句子作为输入，否则使用句子单词列表作为输入。
        - requires_grad: 
        `bool`, True表示更新embedding层参数。
        - vocab_size: 
        `int`， 表示字典库大小。
        - glove_dim:
        `int`，表示通过glove方法得到的单词向量维数。
        - word2id_file: 
        `str`， 存储单词转id的npy文件地址(use_id=False是不需要)。
        - glove_file:
        `str`， 存储预训练向量的npy文件地址。
    - elmo_param:
        - elmo_dim:
        `int`, 表示通过elmo方法得到的单词向量维数。
        - requires_grad: 
        `bool`, True表示更新elmo的参数。
        - elmo_options_file/elmo_weight_file：
        `str`，elmo官方配置文件地址。

---
#### 调用

使用此代码创建`embedder`对象。
```
 self.embedder = Embedder(emb_method=self.opt.emb_method, glove_param=self.opt.glove_param, 
 elmo_param=self.opt.elmo_param, use_gpu=self.opt.use_gpu)
```
- 参数说明
    - emb_method:
    `str`，"elmo"/"glove"/"elom_glove"，表示使用哪种embedding方法。
    - glove_param:
    `dict`，上方介绍的config.py中的字典，当使用glove/elmo_glove生成embedding时需要传入。
    - elmo_param:
    `dict`，上方介绍的config.py中的字典，当使用elmo/elmo_glove生成embedding时需要传入。
    - use_gpu:
    `bool`， 表示是否使用gpu训练。

- 使用说明
```
x = self.embedder(x)
```
- 输入为句子列表，或表示句子维度为(B, L)的Tensor(glove方法`use_id`为`True`时可行)
- 输出为(B, L, f_dim)的Tensor.

-----

### encoder层操作说明

#### 调用
使用此代码创建`encoder`对象

```
# 使用Cnn
self.encoder = Encoder(enc_method=self.opt.enc_method, filters_num=self.opt.filters_num, filters=self.opt.filters, f_dim=self.opt.f_dim)

# 使用rnn/lstm/gru
self.encoder = Encoder(enc_method=self.opt.enc_method, input_size=self.opt.input_size, hidden_size=self.opt.hidden_size, bidirectional=self.opt.bidirectional)
```

- 参数说明
    - cnn
        - enc_method:
        `str`: "cnn"，表示使用cnn做encoder
        - filters_num:
        `int`: 表示卷积核的个数。
        - filters_num:
        `list(int)`: 表示卷积核的长度
        - f_dim:
        `int`: 表示卷积核的宽度，一般是取单词的embedding维度.
    - rnn类
        - enc_method:
        `str`: "rnn"/"gru"/"lstm"， 表示不同的rnn类encoder。
        - input_size:
        `int`: 表示输入待加工向量的维度，在句子层次即为单词的embedding维度。
        - hidden_size:
        `int`: 表示rnn类encoder输出的隐层向量维度.
        - bidirectional:
        `bool`: 表示是双向.
- 使用说明

```
out = self.encoder(inputs, sqe_len)
```
- 输入
    - inputs: 一个(B,L,f_dim)的Tensor
    - sqe_len: 可选，若传入, 自动对输入进行mask.
- 输出
    - cnn
        - len(filters) * (B, filters_num, L)的列表
    - rnn类
        - (B, N/L, hidden_size)的Tensor. (其中，若传入sqe_len, 第二维为:N=max(sqe_len))

------

### attention层操作说明.
#### 调用
使用此代码创建`attenter`对象。
```
self.attenter = Attenter(att_method='Hdot', f_dim, q_dim, q_num)
```
- 参数说明：
    - att_method:
    `str`, 'Hdot', 'Cat', 'Tdot1', 'Tdot2',表示使用何种attention方法, 详细介绍见ppt
    - f_dim:
    `int`, 指表示单词或者句子的向量维度
    - q_dim:
    `int`: query vector的维度
    - q_num:
    `int`: query vector的数量, `Cat`方法中需要

- 使用说明
```
x = self.attenter(W, Q, sqe_len)
```

- 输入
    - W: 为一个(B, L, f_dim)的Tensor
        - 单词层面B代表句子数， L代表句子最大长度， f_dim代表单词向量维数。
        - 句子层面B=1, L代表句子个数， f_dim代表句子向量维数
    - Q：(K, q_dim)的Tensor, 表示query vectors
    - sqe_len: 一个K维tensor, 表示每个句子长度, 不是必须，加上自动对权重矩阵进行mask
- 输出
    - 一个(B, K, f_dim)的Tensor
        - 后两维中的`第i行`代表在`第i个`query vector的attention下生成的向量。

----

## 实验结果
### 测试不同的embedder
- enc_method:`'cnn'`
- att_method:无
 

|glove|elmo|elmo_glove|
|----|----|----|
|0.816|0.781|0.825|


---
### 测试不同encoder:
- emb_method: `'glove'`
- att_method: `'Hdot'` (rnn/gru/lstm)

|cnn|rnn|gru|lstm|
|----|----|----|----|
|0.816| 0.781|0.812|0.820|


---

### 测试不同的attender
- emb_method: `'glove'`
- enc_method: `'lstm'`

|Hdot|Tdot1|Tdot2|Cat|
|----|----|----|----|
| 0.820|0.816|0.799|0.803|

----
### 测试不同的query vector数量
- emb_method:`'glove'`
- enc_method: `'lstm'`
- att_method: `'Hdot'`

|1|2|3|4|
|----|----|----|----|
|0.820|0.803|0.820|0.806|
