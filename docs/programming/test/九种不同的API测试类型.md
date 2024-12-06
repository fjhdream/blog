---
title: 九种不同的API测试类型
categories:
  - Test
tags:
  - 测试
halo:
  site: http://205.234.201.223:8090
  name: 8e9faab7-9b80-44af-a836-340618207795
  publish: true
---
![](http://picbed.fjhdream.cn/202312260919857.png)

## 冒烟测试 (Smoke Testing)

API开发完成后进行此操作。只需验证API是否正常工作，不会出现任何问题。
## 功能测试 (Functional Testing)

根据功能需求创建测试计划，并将结果与预期结果进行比较。
## 集成测试(Integration Testing)

这个测试结合了几个API调用来进行端到端测试。测试了服务内部的通信和数据传输。

## 回归测试(Regression Testing)

此测试确保修复错误或添加新功能时不会破坏现有的API行为。

## 负载测试(Loading Testing)

通过模拟不同的负载来测试应用程序的性能。然后我们可以计算应用程序的容量。

## 压力测试(Stress Testing)

我们故意制造高负载给API，并测试API是否能正常运行。

## UI测试(UI Testing)

这个测试通过与API的UI交互来确保数据能够正确显示。

## 模糊测试(Fuzz Testing)

这将无效或意外的输入数据注入到API中，并尝试使API崩溃。通过这种方式，它可以识别API的漏洞。


> 转载自: https://blog.bytebytego.com/p/ep83-explaining-9-types-of-api-testing