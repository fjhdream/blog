![The three main stages of coding an LLM. This chapter focuses on step 3 of stage 1: implementing the LLM architecture.](https://picbed.fjhdream.cn/202411271656595.png)

## Coding an LLM architecture

![A Gpt Model](https://picbed.fjhdream.cn/202411271658858.png)  
现在，我们将规模扩大到一个小型GPT-2模型的大小，具体而言是指具有1.24亿参数的最小版本，如“语言模型是无监督多任务学习者”中所述。

在深度学习和像GPT这样的大型语言模型的背景下，术语“参数”指的是模型的可训练权重。这些权重实际上是模型的内部变量，在训练过程中被调整和优化，以最小化特定的损失函数。这种优化使模型能够从训练数据中学习。

例如，在一个由2,048 × 2,048维矩阵（或张量）表示的神经网络层中，该矩阵的每个元素都是一个参数。由于有2,048行和2,048列，因此该层的参数总数为2,048乘以2,048，等于4,194,304个参数。

首先， 我们定义一下GPT-2的相关参数

``` python
GPT_CONFIG_124M = { 
	"vocab_size": 50257, # Vocabulary size, BPE tokenizer
	"context_length": 1024, # Context length, maximum number of input tokens
	"emb_dim": 768, # Embedding dimension 
	"n_heads": 12, # Number of attention heads 
	"n_layers": 12, # Number of layers, the number of transformer blocks
	"drop_rate": 0.1, # Dropout rate 
	"qkv_bias": False # Query-Key-Value bias 
}
```

![Dummy GPTModel(The order to code GPT architecture)](https://picbed.fjhdream.cn/202411271716458.png)

## 1. GPT backbone

```python
import torch
import torch.nn as nn

class DummyGPTModel(nn.Module):
def __init__(self, cfg):
	super().__init__()
	self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
	self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
	self.drop_emb = nn.Dropout(cfg["drop_rate"])
	self.trf_blocks = nn.Sequential(
	*[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
	) #1 Uses a placeholder for TransformerBlock
	self.final_norm = DummyLayerNorm(cfg["emb_dim"]) #2 Uses a placeholder for LayerNorm
	self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

def forward(self, in_idx):
	batch_size, seq_len = in_idx.shape
	tok_embeds = self.tok_emb(in_idx)
	pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
	x = tok_embeds + pos_embeds
	x = self.drop_emb(x)
	x = self.trf_blocks(x)
	x = self.final_norm(x)
	logits = self.out_head(x)
	return logits

class DummyTransformerBlock(nn.Module): #3  A simple placeholder class that will be replaced by a real TransformerBlock later
	def __init__(self, cfg):
		super().__init__()

	def forward(self, x): #4 This block does nothing and just returns its input.
		return x

class DummyLayerNorm(nn.Module): #5 A simple placeholder class that will be replaced by a real LayerNorm later
	def __init__(self, normalized_shape, eps=1e-5): #6 The parameters here are just to mimic the LayerNorm interface.
		super().__init__()
	
	def forward(self, x):
		return x
```

## 2. How Data Flow

![how the input data is tokenized, embedded, and fed to the GPT model](https://picbed.fjhdream.cn/202411271726321.png)

``` python
import tiktoken 

tokenizer = tiktoken.get_encoding("gpt2") 
batch = [] 
txt1 = "Every effort moves you" 
txt2 = "Every day holds a" 

batch.append(torch.tensor(tokenizer.encode(txt1))) 
batch.append(torch.tensor(tokenizer.encode(txt2))) 
batch = torch.stack(batch, dim=0) 

print(batch)


### result
tensor([[6109, 3626, 6100, 345],  #1 
		[6109, 1110, 6622, 257]]

torch.manual_seed(123) 
model = DummyGPTModel(GPT_CONFIG_124M) 
logits = model(batch) 
print("Output shape:", logits.shape)
print(logits)

### result
Output shape: torch.Size([2, 4, 50257])   
tensor([[[-1.2034, 0.3201, -0.7130, ..., -1.5548, -0.2390, -0.4667], 
		 [-0.1192, 0.4539, -0.4432, ..., 0.2392, 1.3469, 1.2430], 
		 [ 0.5307, 1.6720, -0.4695, ..., 1.1966, 0.0111, 0.5835], 
		 [ 0.0139, 1.6755, -0.3388, ..., 1.1586, -0.0435, -1.0400]], 
		 
		 [[-1.0908, 0.1798, -0.9484, ..., -1.6047, 0.2439, -0.4530], 
		 [-0.7860, 0.5581, -0.0610, ..., 0.4835, -0.0077, 1.6621], 
		 [ 0.3567, 1.2698, -0.6398, ..., -0.0162, -0.1296, 0.3717], 
		 [-0.2407, -0.7349, -0.5102, ..., 2.0057, -0.3694, 0.1814]]], grad_fn=<UnsafeViewBackward0>)
```

每个文本样本由4个token组成；每个token是一个50,257维的向量，这与Tokenizer的词汇表大小相匹配。

该Embedding有50,257个维度，因为每个维度都对应词汇表中的一个唯一token。当我们实现后处理代码时，我们会将这些50,257维的向量转换回token ID，然后可以将其解码成单词。

现在我们从上到下看了GPT架构及其输入和输出，我们将编写各个占位符，首先从真实的Layer Normalization类开始，这个类将取代之前代码中的DummyLayerNorm。

## 3. Normalizing activations with layer normalization

这个神经网络在学习数据中的潜在模式时有困难，无法达到能够进行准确预测或决策的程度。

现在让我们实现layer normalization来提高神经网络训练的稳定性和效率。Layer normalization的主要思想是调整神经网络层的激活（输出），使其具有均值为0和方差为1的特性，也称为单位方差。这样的调整加速了有效权重的收敛，并确保训练的一致性和可靠性。在GPT-2和现代transformer架构中，layer normalization通常应用在multi-head attention模块的前后，以及我们在DummyLayerNorm占位符中看到的，用于最终输出层之前。  
![how layer normalization functions](https://picbed.fjhdream.cn/202411271736878.png)

``` python
## example code
torch.manual_seed(123) 
batch_example = torch.randn(2, 5)  #1  Creates two training examples with five dimensions (features) each
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU()) 
out = layer(batch_example) 

print(out)

## result
tensor([[0.2260, 0.3470, 0.0000, 0.2216, 0.0000, 0.0000], 
		[0.2133, 0.2394, 0.0000, 0.5198, 0.3297, 0.0000]], grad_fn=<ReluBackward0>)

mean = out.mean(dim=-1, keepdim=True) 
var = out.var(dim=-1, keepdim=True) 
print("Mean:\n", mean) 
print("Variance:\n", var)

## result
Mean: tensor([[0.1324], [0.2170]], grad_fn=<MeanBackward1>) 

Variance: tensor([[0.0231], [0.0398]], grad_fn=<VarBackward0>)
```

> 我们编码的神经网络层由一个线性层和一个非线性激活函数ReLU（即修正线性单元）组成，这是神经网络中的标准激活函数。如果你不熟悉ReLU，它简单地将负输入阈值化为0，从而确保该层只输出正值，这也解释了为什么输出的层没有任何负值。  
> 
> 在像均值或方差计算这样的操作中使用keepdim=True可以确保output tensor保持与input tensor相同的维度数量，即使该操作在通过dim指定的维度上减少了张量。例如，如果不使用keepdim=True，返回的均值张量将是一个二维向量\[0.1324, 0.2170\]而不是一个2 × 1维的矩阵\[\[0.1324\], \[0.2170\]\]。
> 
> dim参数指定了在张量中应执行统计量（此处为均值或方差）计算的维度。

![An illustration of the dim parameter when calculating the mean of a tensor.](https://picbed.fjhdream.cn/202411271744701.png)

让我们对之前获得的层输出应用层归一化。该操作包括减去均值和除以方差的平方根：

``` python
out_norm = (out - mean) / torch.sqrt(var) 
mean = out_norm.mean(dim=-1, keepdim=True) 
var = out_norm.var(dim=-1, keepdim=True) 
print("Normalized layer outputs:\n", out_norm) 
print("Mean:\n", mean) 
print("Variance:\n", var)

## result
Normalized layer outputs: 

tensor([[ 0.6159, 1.4126, -0.8719, 0.5872, -0.8719, -0.8719], 
		[-0.0189, 0.1121, -1.0876, 1.5173, 0.5647, -1.0876]], grad_fn=<DivBackward0>) 

Mean: 

tensor([[-5.9605e-08],
		[1.9868e-08]], grad_fn=<MeanBackward1>) 
		
Variance: 

tensor([[1.], 
		[1.]], grad_fn=<VarBackward0>)
```

让我们实现以上流程到GPT Model中

``` python
class LayerNorm(nn.Module): 
	def __init__(self, emb_dim): 
		super().__init__() 
		self.eps = 1e-5 
		self.scale = nn.Parameter(torch.ones(emb_dim)) 
		self.shift = nn.Parameter(torch.zeros(emb_dim)) 
		
	def forward(self, x): 
		mean = x.mean(dim=-1, keepdim=True) 
		var = x.var(dim=-1, keepdim=True, unbiased=False) 
		norm_x = (x - mean) / torch.sqrt(var + self.eps) 
		return self.scale * norm_x + self.shift
```

使用LayerNorm再之前的Batch Input

``` python
ln = LayerNorm(emb_dim=5) 
out_ln = ln(batch_example)
mean = out_ln.mean(dim=-1, keepdim=True) 
var = out_ln.var(dim=-1, unbiased=False, keepdim=True) 

print("Mean:\n", mean) 
print("Variance:\n", var)

## result
Mean: 

tensor([[ -0.0000], 
		[ 0.0000]], grad_fn=<MeanBackward1>) 

Variance: 

tensor([[1.0000], 
		[1.0000]], grad_fn=<VarBackward0>)
```

![What we have done](https://picbed.fjhdream.cn/202411271755061.png)
