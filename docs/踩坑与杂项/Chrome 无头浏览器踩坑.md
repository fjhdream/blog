> **问题背景:**  
> 使用无头浏览器对网页进行截图时总是发现部分网页图片多时无法加载出来  
> 不管等待多久时间仍然是不会加载出来  
> **问题解决步骤:**
> 1. 取消掉无头浏览器模式后发现可以正常的加载
> 2. 初步怀疑是无头浏览器资源限制问题导致单域名并发数限制
> 3. 但是强制启用h2协议也无法解决问题
> 4. 网上搜索资料发现旧版无头浏览器与有窗口浏览器代码并不是一套导致
> 5. 进而发现Chrome已推出模式, 合并代码
>
> > **问题背景:**  
> 使用无头浏览器对网页进行截图时总是发现部分网页图片多时无法加载出来  
> 不管等待多久时间仍然是不会加载出来  
> **问题解决步骤:**
> 1. 取消掉无头浏览器模式后发现可以正常的加载
>
> 2. 初步怀疑是无头浏览器资源限制问题导致单域名并发数限制
> 3. 但是强制启用h2协议也无法解决问题
> 4. 网上搜索资料发现旧版无头浏览器与有窗口浏览器代码并不是一套导致
> 5. 进而发现Chrome已推出模式, 合并代码  
> **如果你有类似问题不如试试全新无头模式**

# Chrome Headless 模式的更新与改进

在Chrome 112中，Headless模式进行了重大升级[文档](https://developer.chrome.com/docs/chromium/new-headless#whats_new_in_headless)，带来了许多新特性和改进。以下是这次更新的主要内容：

## 新特性介绍

### 统一代码库

新的Headless模式与普通Chrome共享代码库，解决了旧版独立实现带来的许多问题，并简化了维护工作。

### 命令行标志

支持新的命令行标志，包括`--dump-dom`、`--screenshot`、`--print-to-pdf`等，增强了操作和自动化能力。

### 调试功能

通过`--remote-debugging-port`标志，可以对Headless模式下的Chrome实例进行远程调试，极大地方便了开发和调试过程。

## 使用示例

### Puppeteer

使用新的Headless模式非常简单，只需在启动浏览器时指定`headless: 'new'`：

```javascript
const browser = await puppeteer.launch({ headless: 'new' });
```

### Selenium-WebDriver

在Selenium中，可以通过设置Chrome选项来启用新的Headless模式：

``` javascript
const driver = await env.builder().setChromeOptions(options.addArguments('--headless=new')).build();
```

## 总结

这次更新不仅简化了自动化测试的实现，还提高了测试结果的一致性和可靠性。新模式的引入为开发者提供了更强大的工具和更便捷的调试手段。

# Chrome Headless 模式的更新与改进

在Chrome 112中，Headless模式进行了重大升级[文档](https://developer.chrome.com/docs/chromium/new-headless#whats_new_in_headless)，带来了许多新特性和改进。以下是这次更新的主要内容：

## 新特性介绍

### 统一代码库

新的Headless模式与普通Chrome共享代码库，解决了旧版独立实现带来的许多问题，并简化了维护工作。

### 命令行标志

支持新的命令行标志，包括`--dump-dom`、`--screenshot`、`--print-to-pdf`等，增强了操作和自动化能力。

### 调试功能

通过`--remote-debugging-port`标志，可以对Headless模式下的Chrome实例进行远程调试，极大地方便了开发和调试过程。

## 使用示例

### Puppeteer

使用新的Headless模式非常简单，只需在启动浏览器时指定`headless: 'new'`：

```javascript
const browser = await puppeteer.launch({ headless: 'new' });
```

### Selenium-WebDriver

在Selenium中，可以通过设置Chrome选项来启用新的Headless模式：

``` javascript
const driver = await env.builder().setChromeOptions(options.addArguments('--headless=new')).build();
```

## 总结

这次更新不仅简化了自动化测试的实现，还提高了测试结果的一致性和可靠性。新模式的引入为开发者提供了更强大的工具和更便捷的调试手段。

