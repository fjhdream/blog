
[SpringDoc](https://docs.spring.io/spring-boot/docs/current/reference/html/howto.html#howto.webserver.configure-ssl)

## 概述
1. 获取JKS文件
2. Spring application.properties
## 获取JKS文件
因为博主机器域名在华为云上管理, 所以这里以华为云为例, 其他云厂商应该也大同小异

1. 创建SSL证书![](./attachments/Pasted%20image%2020230528133827.png)
2. 下载配置文件
![](./attachments/Pasted%20image%2020230528133908.png)
3. 文件内容如下
![](./attachments/Pasted%20image%2020230528134231.png)

![](./attachments/Pasted%20image%2020230528134251.png)这里就有我们需要的jks文件了, 另一个是jks的解析密钥
. 我们将jks文件移动到Spring项目中去

## 配置SpringSSL
1. 放置jks文件到对应目录(目录可以自定义) ![](./attachments/Pasted%20image%2020230528134448.png)
2. 配置application.properties ![](./attachments/Pasted%20image%2020230528134548.png)
3. 启动项目 ![](./attachments/Pasted%20image%2020230528134623.png)这里日志显示已经开启https了