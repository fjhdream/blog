---
title: langchain相关概念
categories:
  - AI
tags:
  - langchain
halo:
  site: http://205.234.201.223:8090
  name: 31aaac37-0ad2-4841-ad03-99d3d41b6164
  publish: true
---


**概念总览**
![langchain相关概念 2024-01-10_17.56.05.excalidraw](http://picbed.fjhdream.cn/202401101759550.svg)

> **原文地址**: https://chat.openai.com/g/g-cVPwtQBU1-softenginner/c/27c9e7e7-e225-44c4-81c3-3eefbe5ebdc2
# Models
LangChain集成的模型主要有两种类型：LLMs和Chat Models。它们的输入和输出类型决定了它们的定义。
## LLM
LangChain中的LLMs指的是纯文本补全模型。它们封装的API接受一个字符串提示作为输入，并输出一个字符串补全。OpenAI的GPT-3是一个LLM。
## Chat Models
聊天模型通常由LLM支持，但专门用于进行对话。关键是，它们的提供者API使用与纯文本补全模型不同的接口。它们不是接受单个字符串作为输入，而是接受一个聊天消息列表，并返回一个AI消息作为输出。有关消息的具体内容
### Messages

ChatModels接受消息列表作为输入并返回一条消息。有几种不同类型的消息。所有消息都有 `role` 和 `content` 属性。 `role` 描述了谁在说这条消息。LangChain为不同角色提供了不同的消息类。 `content` 属性描述了消息的内容

- String  一个字符串
- Dictionary 一个字典  (多模态输入)
#### HumanMessage
这代表用户的一条消息。通常只包含内容。
#### AIMessage
代表了一个来自模型的消息。其中可能包含 `additional_kwargs` - 例如，如果使用OpenAI函数调用，则可能包含 `functional_call` 。
#### SystemMessage
代表一个系统消息。只有一些模型支持这个。这告诉模型如何行为。这通常只包含内容。
#### FunctionMessage
代表了一个函数调用的结果。除了 `role` 和 `content` 之外，这个消息还有一个 `name` 参数，它传达了产生这个结果的函数的名称
#### ToolMessage
代表了一个工具调用的结果。这与FunctionMessage不同，以匹配OpenAI的 `function` 和 `tool` 消息类型。除了 `role` 和 `content` 之外，这个消息还有一个 `tool_call_id` 参数，用于传达调用产生此结果的工具的id。

## 两种模型比较

这两种API类型的输入和输出模式非常不同。这意味着与它们交互的最佳方式可能会有很大的不同。虽然LangChain使得它们可以互换使用，但并不意味着你应该这样做。特别是，对于LLMs和ChatModels的提示策略可能会有很大的不同。这意味着你需要确保你使用的提示是为你正在使用的模型类型设计的。

# Prompts

语言模型的输入通常被称为提示。通常情况下，您的应用程序中的用户输入不是直接输入到模型中的。相反，他们的输入会以某种方式进行转换，以产生进入模型的字符串或消息列表。将用户输入转换为最终字符串或消息的对象被称为“提示模板”。LangChain提供了几种抽象方法，使得处理提示更加容易。

## PromptValue
ChatModels和LLMs采用不同的输入类型。PromptValue是一个设计用于两者之间互操作的类。它公开了一个方法可以转换为字符串（与LLMs一起使用），另一个方法可以转换为消息列表（与ChatModels一起使用）。

## PromptTemplate

提示模板的示例。它由一个模板字符串组成。然后，使用用户输入对该字符串进行格式化，生成最终字符串。

## MessagePromptTemplate
这是一个提示模板的示例。它包括一个模板消息 - 意味着一个特定的角色和一个提示模板。然后，将使用用户输入对这个提示模板进行格式化，生成一个最终字符串，成为该消息的 `content` 。
#### HumanMessagePromptTemplate
#### AIMessagePromptTemplate
#### SystemMessagePromptTemplate

## MessagesPlaceholder
提示的输入可以是一系列的消息。这时候你会使用一个MessagesPlaceholder。这些对象由一个 `variable_name` 参数参数化。与这个 `variable_name` 值相同的输入应该是一个消息列表。

## ChatPromptTemplate
提示模板的示例。它由MessagePromptTemplates或MessagePlaceholders列表组成。然后，使用用户输入进行格式化，生成最终的消息列表。

# Output Parsers
模型的输出要么是字符串，要么是一条消息。通常，字符串或消息中包含以特定格式进行格式化的信息，以便在下游使用（例如逗号分隔的列表或JSON数据块）。输出解析器负责接收模型的输出并将其转换为更可用的形式。这些解析器通常处理输出消息的 `content` 字段，但有时也会处理 `additional_kwargs` 字段中的值。

## StrOutputParser
这是一个简单的输出解析器，它只是将语言模型（LLM或ChatModel）的输出转换为字符串。如果模型是一个LLM（因此输出一个字符串），它只是将该字符串传递。如果输出是一个ChatModel（因此输出一个消息），它会传递消息的 `.content` 属性。

## OpenAI Functions Parsers
有一些专门用于处理OpenAI函数调用的解析器。它们使用 `function_call` 和 `arguments` 参数的输出（这些参数位于 `additional_kwargs` 内部）并对其进行处理，主要忽略内容。

## Agent Output Parsers
代理是使用语言模型来确定采取哪些步骤的系统。因此，语言模型的输出需要被解析成一些能够表示应采取的行动（如果有的话）的模式。AgentOutputParsers负责将原始的LLM或ChatModel的输出转换成该模式。这些输出解析器内部的逻辑可能因所使用的模型和提示策略而有所不同。

# 示例

>  推荐一家国内可以使用的的 API 中转网站CloseAI, 完美兼容开发框架的 API : 
>   https://referer.shadowai.xyz/r/1000886

``` python
from langchain.chat_models import ChatOpenAI  
from langchain.prompts import ChatPromptTemplate  
from langchain.schema import StrOutputParser  
from langchain.schema.runnable import RunnablePassthrough  
  
prompt = ChatPromptTemplate.from_template(  
    "Tell me a short joke about {topic}"  
)  
output_parser = StrOutputParser()  
model = ChatOpenAI(model="gpt-3.5-turbo")  
chain = (  
        {"topic": RunnablePassthrough()}  
        | prompt  
        | model  
        | output_parser  
)  
  
if __name__ == "__main__":  
    print(chain.batch(["ice cream", "spaghetti", "dumplings"]))
```