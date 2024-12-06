---
title: 深入浅出 langchain 1.prompt 与 model
categories:
  - AI
tags:
  - langchain
halo:
  site: http://205.234.201.223:8090
  name: a18efce2-4f81-4699-b343-1962056837bf
  publish: false
---
## 示例

从代码入手来看原理
``` python
from langchain_core.output_parsers import StrOutputParser  
from langchain_core.prompts import ChatPromptTemplate  
from langchain_openai import ChatOpenAI  
  
prompt = ChatPromptTemplate.from_template("tell me a short joke about {topic}")  
model = ChatOpenAI(model="gpt-4")  
output_parser = StrOutputParser()  
  
chain = prompt | model | output_parser  
  
chain.invoke({"topic": "ice cream"})
```

> chain = prompt | model | output_parser

`|`是 Unix 管道操作符, 将不同的组件链接到一起, 一组组件的输出作为下一组件的输入.

## Prompt

`prompt` 是一个 `BasePromptTemplate` ，这意味着它接收一个模板变量的字典并生成一个 `PromptValue` 。一个 `PromptValue` 是一个完成提示的包装器，可以传递给 `LLM` （接受字符串作为输入）或 `ChatModel` （接受消息序列作为输入）。它可以与任何语言模型类型一起工作，因为它定义了生成 `BaseMessage` 和生成字符串的逻辑。

以下是 PromptValue 的输入
``` python
prompt_value = prompt.invoke({"topic": "ice cream"})

prompt_value
# ChatPromptValue(messages=[HumanMessage(content='tell me a short joke about ice cream')])

prompt_value.to_messages()
# [HumanMessage(content='tell me a short joke about ice cream')]

prompt_value.to_string()
# 'Human: tell me a short joke about ice cream'
```

## Model

然后将 `PromptValue` 传递给 `model` 。在这种情况下，我们的 `model` 是一个 `ChatModel` ，意味着它将输出一个 `BaseMessage` 。

``` python
message = model.invoke(prompt_value)  

message
# AIMessage(content="Why don't ice creams ever get invited to parties?\n\nBecause they always bring a melt down!")
```


如果我们的 `model` 是一个 `LLM` ，它会输出一个字符串。

``` python
from langchain_openai.llms import OpenAI  
  
llm = OpenAI(model="gpt-3.5-turbo-instruct")  
llm.invoke(prompt_value)

# '\n\nRobot: Why did the ice cream truck break down? Because it had a meltdown!'
```

## Output parser

最后，我们将我们的 `model` 输出传递给 `output_parser` ，这是一个 `BaseOutputParser` ，它接受字符串或 `BaseMessage` 作为输入。这个 `StrOutputParser` 特别简单地将任何输入转换为字符串

``` python
output_parser.invoke(message)

# "Why did the ice cream go to therapy? \n\nBecause it had too many toppings and couldn't find its cone-fidence!"
```

## Summary

运行流程图如下: 

![运行流程图](http://picbed.fjhdream.cn/202402271704387.svg)

我们将用户输入变成字典后, 传递给 PromptTemplate 包装成 PromptValue, 传递给 ChatModel 后, Model 给我们返回 ChatMessage, 再将其传递给 StrOutputParser, 最终解析成 String 类型