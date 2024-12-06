## 示例

还是从一个简单的代码示例开始

```python
from langchain.agents import create_tool_calling_agent  
  
agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
```

我们可以从这个示例里面看到通过了`create_tool_calling_agent`方法创建出来了一个agent, 传递了三个参数:

1. llm 我们使用的大模型
2. tools 可以使用的工具, 也可以认为就是functions,函数
3. prompt 提示词, 这个没有什么需要补充的

传递的tools有个别名, function_calls

那我们就会有个疑问什么是tool?

## Tool

我们可以看下官方解释:

Tools are interfaces that an agent, chain, or LLM can use to interact with the world. They combine a few things:

1. The name of the tool
2. A description of what the tool is
3. JSON schema of what the inputs to the tool are
4. The function to call
5. Whether the result of a tool should be returned directly to the user

其实跟我们平常理解的函数是很像的, 需要描述Tool的名称 作用 输入 输出.  
可以认为这Tool 其实就是callback, 当AI认为需要使用这个方法的时候, 就会调用到这里

### 创建Tool

```python
@tool  
def search(query: str) -> str:  
"""Look up things online."""  
return "LangChain"
```

有了Tool 的含义解释, 那么也就很容易理解agent的含义了

## Agent

Agent就像是一个推理的任务中心, 当你咨询某个问题的时候, 他会推理这个问题是否需要借助某些工具(Tool)来实现, 如果需要就会调用到这个Tool

一个最简单的例子就是搜索功能, 绕过LLM的知识库限制

``` python
from langchain.agents import AgentExecutor, create_tool_calling_agent  
from langchain_community.tools.tavily_search import TavilySearchResults  
from langchain_core.prompts import ChatPromptTemplate  
  
tools = [TavilySearchResults(max_results=1)]
prompt = ChatPromptTemplate.from_messages(  
[  
(  
"system",  
"You are a helpful assistant. Make sure to use the tavily_search_results_json tool for information.",  
),  
("placeholder", "{chat_history}"),  
("human", "{input}"),  
("placeholder", "{agent_scratchpad}"),  
]  
)  
  
# Construct the Tools agent  
agent = create_tool_calling_agent(llm, tools, prompt)

# Create an agent executor by passing in the agent and tools  
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)  
agent_executor.invoke({"input": "what is LangChain?"})
```