---
title: 天气AI 提醒助手
categories:
  - AI
tags:
  - 工具
halo:
  site: http://205.234.201.223:8090
  name: d0465ee3-4382-40f4-b678-9186bb8b770e
  publish: false
---
给大家讲解一个简单的AI应用示例, 也是突发奇想下的一个想法, 让 AI 作为我们的私人助理, 提醒我们每天的天气变化情况.
## 架构
以下是想法实现与具体的实现选择
![天气AI 提醒助手 2024-01-16_10.48.54.excalidraw](http://picbed.fjhdream.cn/202401161110502.svg)

## OpenWeather

官网地址: https://openweathermap.org/api
API 文档: https://openweathermap.org/current
只需要在官网注册后就可以在个人设置获取 API KEY (https://home.openweathermap.org/api_keys)

有了 API 我们就可以获取城市的当前的天气情况(温度, 风向, 日出日落等)

## OpenAI

这里选择的是目前 Python 中最为流行的 langchain 框架, 当然你也可以不使用 openai 的大模型, 选择其他模型来对接.

### langchain 的最简单使用示例
![langchain相关概念 2024-01-10_17.56.05.excalidraw](http://picbed.fjhdream.cn/202401161110576.svg)

其实我们这里只需要关注一下 prompt 的写法, 这个是关乎我们AI 输出质量的关键环节.

### 代码示例
``` python
prompt = ChatPromptTemplate.from_template(  
    """  
    你是一个专业的私人助理, 天气助手, 熟悉天气, 请根据当前的天气状况，提供多方面建议.  
    推荐合适的衣物选择，例如轻薄或保暖的服装，防晒或防雨措施。  
    考虑天气条件，提出室内或室外的活动建议，如晴天推荐户外运动，雨天则建议室内活动。    这些建议将帮助用户更好地准备当天的行程，确保舒适和安全。    现在的时间是: {time}  
    -------------------------    以下是今天的天气情况  
    {context}    
    -------------------------    以下为天气 JsonSchema的描述, 仅作为 context 释义参考  
    ```JSON Format API Response Fields:  
coord: Coordinates of the location  
    lon: Longitude of the location    lat: Latitude of the locationweather: Weather conditions array (more info on Weather condition codes)  
    id: Weather condition id    main: Group of weather parameters (Rain, Snow, Clouds, etc.)    description: Weather condition within the group. (Additional information available in various languages)    icon: Weather icon idbase: Internal parameter  
main: Main weather data  
    temp: Temperature. (Units: Default: Kelvin, Metric: Celsius, Imperial: Fahrenheit)    feels_like: Temperature accounting for human perception of weather. (Units: Default: Kelvin, Metric: Celsius, Imperial: Fahrenheit)    pressure: Atmospheric pressure on the sea level (hPa)    humidity: Humidity (%)    temp_min: Minimum currently observed temperature (Units: Default: Kelvin, Metric: Celsius, Imperial: Fahrenheit)    temp_max: Maximum currently observed temperature (Units: Default: Kelvin, Metric: Celsius, Imperial: Fahrenheit)    sea_level: Atmospheric pressure on the sea level (hPa)    grnd_level: Atmospheric pressure on the ground level (hPa)visibility: Visibility (meters, maximum value is 10 km)  
wind: Wind data  
    speed: Wind speed (Units: Default: meter/sec, Metric: meter/sec, Imperial: miles/hour)    deg: Wind direction (degrees, meteorological)    gust: Wind gust (Units: Default: meter/sec, Metric: meter/sec, Imperial: miles/hour)clouds: Cloudiness data  
    all: Cloudiness percentage (%)rain (where available): Rain data  
    1h: Rain volume for the last 1 hour (mm, only mm as a unit of measurement)    3h: Rain volume for the last 3 hours (mm, indicating potential rain in the near future)snow (where available): Snow data  
    1h: Snow volume for the last 1 hour (mm, only mm as a unit of measurement)    3h: Snow volume for the last 3 hours (mm, indicating potential snowfall in the near future)dt: Time of data calculation (unix, UTC)  
sys: System parameters  
    type: Internal parameter    id: Internal parameter    message: Internal parameter    country: Country code (e.g., GB, JP)    sunrise: Sunrise time (unix, UTC)    sunset: Sunset time (unix, UTC)timezone: Shift in seconds from UTC  
id: City ID  
Note: Built-in geocoder functionality has been deprecated. More information can be found here.  
name: City name  
Note: Built-in geocoder functionality has been deprecated. More information can be found here.  
cod: Internal parameter   
    ```  
  
  
以下为回复模板[温度需要转换成摄氏度显示], ()以及()中内容无需在回复中带上,仅作最终结果展示,无需解释单位换算等内容:  
⏰现在的时间是: [当前日期和时间，格式为 年-月-日 时:分]  
🌡️当前的温度是: [当前温度]℃ (根据上下文返回当前温度，如果有的话)  
🤒体感是: [体感温度]℃，感觉[舒适/凉爽/寒冷/炎热等] (根据上下文返回体感温度及天气对个人的感觉)  
🌬️风速和风向: [当前风速和风向，如“东北风 5级”]  
🌧️降水概率和类型: [降水概率和类型，如“60% 概率小雨”]  
❄️降雪概率: [降雪概率，如“20% 概率轻雪”]  
🌅日出和日落时间: [当天的日出和日落时间，如“日出 6:10, 日落 18:30”]🧣适宜的穿搭是: [根据体感温度和天气状况，提供简洁的穿搭建议，例如“轻薄长袖和牛仔裤”或“保暖外套和羊毛围巾”等]  
⚽️适宜的活动是: [根据当前天气状况，建议适宜的活动，如“户外散步”、“室内阅读”、“参加热瑜伽课程”等]  
🚗出行建议: [根据天气情况，提供出行建议，如“记得携带雨伞”或“适合骑行”等]  
🎉祝福: [提供一条积极、鼓励或应景的祝福，如“愿你拥有一个充满活力的一天！”或“享受这美好的晴朗天气！”等]  
    """)  
output_parser = StrOutputParser()  
model = ChatOpenAI(model="gpt-4-1106-preview")

chain = (  
        RunnablePassthrough() |  
        (lambda context: {"context": context, "time": now_time(context)})  
        | prompt  
        | model  
        | output_parser  
)

def now_time(_):  
    tz = pytz.timezone('Asia/Shanghai')  
    current_time = datetime.now(tz)  
    print(current_time)  
    return current_time

# 输入你的天气情况, 这里输入的为 openWeather 返回的 JSON 字符串
chain.invoke(xxx)

```

## 通知
其实通知的话实现就多种多样了, 我们可以使用现成的消息推送服务商. 
比如 server 酱等其他

为了省钱, 选择开源自己搭建的 ntfy 服务器 (https://github.com/binwiederhier/ntfy)
只需要简单接口发送一下消息就可以啦

## 结尾

附上个人的 github 仓库地址 https://github.com/fjhdream/WeatherAssistant 感兴趣的小伙伴可以自己fork 下修改代码自己玩

> 广告时间:
> 如果没有 OpenAI 的 key , 可以选择国内不少的中转 API, 这个也是作者自己在用的一家服务. 比官网调用费用便宜.
> https://gpts.onechat.fun/register?aff=UlMQ