翻译器全局架构蓝图 (Module & Class 级别)



### 代码过程
1. 处理语料， 将语料读取为 jsonl 格式，英文和中文一一对应  
  * assemble_data.py 语料处理代码 -> translation_dataset.jsonl
2. 处理 tokenizer , 单一功能， 将句子输出为 ids 序列， 切处理为等长
   * tokenizer.py
3. 处理基础表示层, （1）嵌入矩阵 （2） 位置矩阵 最终得到 有位置信息和语义信息的词向量
  * represent_layer.py
4. 多头注意力模型
   mutil-head-attention
5.  编码器层 EncoderBlock， 多头注意力机制 + 前馈神经网络
  * x -> attention -> Add 残差链接 -> layerNorm 层归一化
  * encoder_layer.py 
6.  将 EncoderBlock 堆叠 N 层， 
  * encoder.py
7. 解码层 decoder-layer
   * 掩码自注意力层， 让解码器理解自已已经生成的内容，但是不看未来
   * 交叉注意力层， 这里是真正和编码器交互的地方
   * 前馈神经网络层 对融合了原文和译文信息的特则进行非线性加工
8. 解码器 decoder 堆叠 N 层 
  * decoder.py 最终还要经过一层映射， 转化为整个词表的概率， 为了后续方便进行loss 计算
9. 到这里模型 model 就结束了，接下来
   * 训练的处理





阶段一：数据与预处理模块 (Data & Preprocessing)
说明：将真实文本转化为模型可识别的张量，并生成关键的遮罩（Mask）。
● 调用 Tokenizer，将中英文本转化为 ID 序列。
● 辅助函数 create_masks(src_ids, tgt_ids)：
  ○ 输入：源语言 ID 和目标语言 ID。
  ○ 核心逻辑：生成 src_mask（遮蔽原文中的 PAD）和 tgt_mask（极度关键：不仅要遮蔽 PAD，还要生成下三角矩阵 look_ahead_mask，防止解码器在训练时“偷看”未来的英文单词）。


阶段二：基础表示层 (Representation Layer)
说明：负责把离散的词 ID 变成富有语义和位置信息的稠密向量。
● TokenEmbedding(nn.Module)：
  ○ 核心逻辑：封装 nn.Embedding，但在输出时需要乘以 $\sqrt{d_{model}}$ 进行数值缩放（Transformer 论文的细节规范）。
● PositionalEncoding(nn.Module)：
  ○ 核心逻辑：由于 Attention 是无视顺序的，必须纯手工实现正弦和余弦函数公式，将位置信息（绝对位置）以加法形式注入到词向量中。


阶段三：核心计算引擎 (The Attention Engine)
说明：整个模型的心脏，负责计算词与词之间的关联度。
● MultiHeadAttention(nn.Module)：
  ○ 输入：Q, K, V 张量，以及 Mask。
  ○ 内部组件：4 个 nn.Linear 层（分别对应 $W^Q, W^K, W^V, W^O$）。
  ○ 核心逻辑：
    ⅰ. 使用 view 和 transpose 将维度拆分为多头 [Batch, Head, SeqLen, D_k]。
    ⅱ. 手写缩放点积公式：$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$。
    ⅲ. 处理 Mask 逻辑（将 Mask 为 0 的地方替换为 -1e9）。
    ⅳ. 拼接多头并经过 $W^O$ 线性投影输出。


阶段四：网络积木块 (Transformer Blocks)
说明：组装 Attention、前馈网络（FFN）、残差连接和层归一化。
● PositionwiseFeedForward(nn.Module)：
  ○ 核心逻辑：实现两个线性层，中间夹一个非线性激活函数（通常是 ReLU 或 GELU），用于在每个位置上独立地升维再降维，增加非线性拟合能力。
● EncoderLayer(nn.Module)：
  ○ 内部组件：1 个 MultiHeadAttention，1 个 FeedForward，2 个 nn.LayerNorm。
  ○ 核心逻辑：实现经典的结构：x = LayerNorm(x + Sublayer(x))。
● DecoderLayer(nn.Module)：
  ○ 内部组件：2 个 MultiHeadAttention（一个是 Masked Self-Attention，另一个是 Cross-Attention），1 个 FeedForward，3 个 nn.LayerNorm。
  ○ 核心逻辑：重点处理 Cross-Attention，确保接收来自 Encoder 的 Memory 作为 K 和 V，而自己的隐状态作为 Q。

阶段五：组装宏观模型 (The Macro Model)
说明：将所有积木堆叠起来，形成完整的 Seq2Seq 架构。
● Encoder(nn.Module)：
  ○ 核心逻辑：包含 TokenEmbedding、PositionalEncoding，并使用 nn.ModuleList 串联 $N$ 层 EncoderLayer。
● Decoder(nn.Module)：
  ○ 核心逻辑：与 Encoder 类似，但串联的是 $N$ 层 DecoderLayer。
● Transformer(nn.Module)：
  ○ 内部组件：Encoder、Decoder、一个最终的生成器 Generator (nn.Linear)。
  ○ 核心逻辑：串接整个前向传播（Forward Pass）。将 Decoder 的输出维度映射回英文词表的大小，以便后续通过 Softmax 计算概率。


阶段六：训练与推理控制流 (Training & Inference Loops)
说明：让模型动起来的外围逻辑。
● train_epoch(...)：
  ○ 核心逻辑：标准的 PyTorch 训练循环。喂入数据，计算 CrossEntropyLoss（注意要忽略 PAD token 的 loss），反向传播，优化器步进。
● greedy_decode(...)：
  ○ 核心逻辑：自回归（Autoregressive）推理函数。输入中文句子，给 Decoder 喂入 <SOS> 启动符，通过 while 循环一个词一个词地预测，直到输出 <EOS> 或达到最大长度。
这就是你将要亲手构建的完整翻译器帝国。






