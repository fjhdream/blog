---
title: 使用 CertBot 配置管理 SSL 证书
categories:
  - 技术学习
tags:
  - 教程
  - 工具
halo:
  site: http://205.234.201.223:8090
  name: f5e49919-686a-4acd-be48-5f6303b32a2b
  publish: true
---
本文将介绍如何使用Certbot，一个自动化的SSL/TLS证书工具，来管理和维护Nginx服务器的证书。

## 什么是Certbot？

Certbot是一个由[EFF](https://eff.org/)（电子前哨基金会）支持的免费工具，它能够轻松地为网站自动获取、部署和续订[Let's Encrypt](https://letsencrypt.org/)提供的免费SSL/TLS证书。Certbot的主要目的是简化证书的获取和维护过程，使HTTPS部署变得简单快捷。

## 为什么选择Certbot？

1. **自动化**: 自动完成证书的申请、更新和部署。
2. **免费**: Let's Encrypt证书完全免费。
3. **安全**: 定期自动更新证书，确保安全性。
4. **社区支持**: 由EFF和庞大的社区提供支持。

## 安装和配置Certbot

在开始之前，确保你拥有一个有效的域名，并且这个域名已经解析到你的服务器上。

### 1. 安装Certbot

根据你的操作系统选择合适的安装指令。

**对于Debian/Ubuntu:**

``` bash
sudo apt update 
sudo apt install certbot python3-certbot-nginx
```

**对于CentOS/RHEL:**

``` bash
sudo yum install epel-release 
sudo yum install certbot python3-certbot-nginx
```

### 2. 获取SSL/TLS证书

运行以下命令，Certbot将自动为你的网站获取证书并配置Nginx。

``` bash
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com
```
替换`yourdomain.com`为你的实际域名。

### 3. 自动续订证书

Let's Encrypt颁发的证书有效期为90天，但Certbot会自动续订这些证书。你可以通过以下命令测试自动续签是否成功：

``` bash
sudo certbot renew --dry-run
```

### 4. 验证HTTPS

在浏览器中输入你的网站地址，查看是否启用了HTTPS并且证书是否正确安装。

## 最佳实践和注意事项

- **定期检查**: 虽然Certbot会自动续订证书，但定期检查网站的HTTPS状态是一个好习惯。
- **安全配置**: 确保Nginx的其它安全设置也得到妥善配置，比如使用强密码、定期更新软件。
- **备份**: 定期备份Nginx配置和SSL证书，以防不测。

## 结论

Certbot是一个强大且易用的工具，它能帮助你自动管理SSL/TLS证书，确保网站安全。通过本文的介绍和教程，你应该能够顺利地使用Certbot为你的Nginx服务器配置SSL证书。记住，安全是一个持续的过程，持续关注并更新你的配置和知识，才能更好地保护你的网站和用户。