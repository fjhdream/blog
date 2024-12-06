
[SpringDoc](https://docs.spring.io/spring-boot/docs/current/reference/html/howto.html#howto.webserver.configure-ssl)

## 概述
1. 获取JKS文件
2. Spring application.properties
## 获取JKS文件
因为博主机器域名在华为云上管理, 所以这里以华为云为例, 其他云厂商应该也大同小异

1. 创建SSL证书![](http://picbed.fjhdream.cn/202312121402487.png)
2. 下载配置文件
![](http://picbed.fjhdream.cn/202312121402510.png)
3. 文件内容如下
![](http://picbed.fjhdream.cn/202312121402521.png)

![](http://picbed.fjhdream.cn/202312121402533.png)这里就有我们需要的jks文件了, 另一个是jks的解析密钥
. 我们将jks文件移动到Spring项目中去

## 配置SpringSSL
1. 放置jks文件到对应目录(目录可以自定义) ![](http://picbed.fjhdream.cn/202312121402546.png)
2. 配置application.properties ![](http://picbed.fjhdream.cn/202312121402556.png)
3. 启动项目 ![](http://picbed.fjhdream.cn/202312121402567.png)这里日志显示已经开启https了