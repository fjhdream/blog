---
title: 深入浅出 langchain 2. RAG
categories:
  - AI
tags:
  - langchain
halo:
  site: http://205.234.201.223:8090
  name: 19641bdb-d30a-4622-b064-69c65c908700
  publish: false
---
## 概念
"RAG" 指的是 "Retrieval-Augmented Generation"，这是一个在自然语言处理和人工智能领域中的概念，特别是在生成型任务中（如文本生成、聊天机器人等）。

RAG 通过结合检索（Retrieval）和生成（Generation）两种技术，来提高模型的性能和输出的质量。具体来说，RAG 首先从一个大型的文档集合中检索出与输入查询相关的信息，然后将这些信息作为上下文输入到一个生成模型（如 GPT）中，以生成更准确、更丰富的回答或内容。

## 示例

``` python
from langchain_community.vectorstores import DocArrayInMemorySearch  
from langchain_core.output_parsers import StrOutputParser  
from langchain_core.prompts import ChatPromptTemplate  
from langchain_core.runnables import RunnableParallel, RunnablePassthrough  
from langchain_openai.chat_models import ChatOpenAI  
from langchain_openai.embeddings import OpenAIEmbeddings  
  
vectorstore = DocArrayInMemorySearch.from_texts(  
["harrison worked at kensho", "bears like to eat honey"],  
embedding=OpenAIEmbeddings(),  
)  
retriever = vectorstore.as_retriever()  
  
template = """Answer the question based only on the following context:  
{context}  
  
Question: {question}  
"""  
prompt = ChatPromptTemplate.from_template(template)  
model = ChatOpenAI()  
output_parser = StrOutputParser()  
  
setup_and_retrieval = RunnableParallel(  
{"context": retriever, "question": RunnablePassthrough()}  
)  
chain = setup_and_retrieval | prompt | model | output_parser  
  
chain.invoke("where did harrison work?")
```

观察上述 chain 的核心链条

> chain = setup_and_retrieval | prompt | model | output_parser

我们多出来一个处理过程`setup_and_retrieval`, 观察上述代码这个过程又涉及几个额外的概念`vectorstore`, `embedding`, `retriever`, 后面会对这些概念进行一一拆解

先看下`retriever`, 不单单是可以使用上述方法, 可以使用直接 invoke 进行调用
``` python
retriever.invoke("where did harrison work?")
```

其实完整的链条就是
``` python
setup_and_retrieval = RunnableParallel(  
{"context": retriever, "question": RunnablePassthrough()}  
)  
chain = setup_and_retrieval | prompt | model | output_parser
```

用流程图具体表示的话就是:

![](http://picbed.fjhdream.cn/202402271720678.svg)

到这里的话我们还有一些概念没有了解, 我们来倒叙讲解

## Retriver

英文解释:
>  "retriever" 一词最早出现在17世纪。它由动词"retrieve"（取回）派生而来，最初指代专门训练用于捡拾打倒的游戏鸟类的狗。后来逐渐演变为特指这类品种的犬只。

检索器是一种接口，它根据非结构化的查询返回文档。它比矢量存储更通用。检索器不需要能够存储文档，只需能够返回（或检索）它们。矢量存储可以用作检索器的支撑，但还有其他类型的检索器。

检索器接受一个字符串查询作为输入，并将 `Document` 列表作为输出返回

用一个普适的例子来类比: 这个就相当于我们通过 retriver(寻回犬), 给他一些信号(字符串), 他能够帮我们取回想要的东西(文档)

再回到上述代码中, retiever 是如何生成的:

``` python
retriever = vectorstore.as_retriever() 
```

## Vector Store

 向量存储: 
 Vector Store，或称向量数据库，是一种专门用于存储和检索向量数据的数据库。在人工智能和机器学习领域，尤其是在自然语言处理（NLP）和图像识别等应用中，数据通常会被转换成高维向量形式。

存储和检索非结构化数据的最常见方式之一是将其嵌入并存储结果向量，然后在查询时嵌入非结构化查询并检索与嵌入查询“最相似”的嵌入向量。向量存储库负责存储嵌入的数据并为您执行向量搜索。

用一个普适的例子来类比:  我们人对世界存在的概念都有自己的理解, 比如: 我现在说红色的大花袄, 我们脑海中就会出现相关的画面以及物品. 比如: 茂密的森林, 脑海中就会出现一篇森林的景象.        这里的红色的大花袄, 茂密的森林 对于向量数据库来说就是一组向量(\[0.01, 0.12,0.32\]), 通过这组向量获取到的数据就是向量数据库查询出来的东西(也就是我们脑海中浮现的画面)


再回到代码中来看下

``` python
vectorstore = DocArrayInMemorySearch.from_texts(  
["harrison worked at kensho", "bears like to eat honey"],  
embedding=OpenAIEmbeddings(),  
) 
```
这里就是生成一个内存存储的向量数据库, 将一些字符串内容存储到向量数据库中.
那么 embedding 是什么呢?

## Embedding

Embedding类是一个用于与文本嵌入模型交互的类。有许多嵌入模型提供商（OpenAI，Cohere，Hugging Face等）-这个类旨在为所有这些提供标准的接口。

Embedding创建了文本的向量表示。这很有用，因为这意味着我们可以在向量空间中考虑文本，并执行一些操作，例如语义搜索，在向量空间中寻找最相似的文本片段。

简单理解就是: 
有个模型能将文本输入变成一组向量方便存储分析中, 当输入文本后, 也可以分析这些文本来找到相关的一些文本.