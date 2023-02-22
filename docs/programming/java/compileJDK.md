# 编译JDK
[官网build文档](https://openjdk.org/groups/build/doc/building.html)
## 编译步骤
本机编译环境:  
- CPU: Apple M1 Max
- JDK: oracle JDK 17 On arm64
- 编译JDK版本: JDK18  tag: jdk-18+22

1.  获取源代码
	`git clone https://git.openjdk.java.net/jdk/`
2. 编译环境和依赖环境
	`bash configure`
3. 编译
	`make images`
4. 验证
	`./build/*/images/jdk/bin/java -version`
5. 基本测试
	`make run-test-tier1`

## 编译踩坑记
###  checking whether the C compiler works... no
原因: 因为本机macos是从Intel 的MacBook 迁移到M1 的MacBook
通过重新安装XCode 与 XCode-Tools 来重新安装Clang等工具解决

[参考来源](https://developer.apple.com/forums/thread/112515)
``` bash
I had clang updated by the following steps:

rm -rf /Applications/Xcode.app  
rm -rf /Library/Developer/CommandLineTools  
rm -rf $HOME/Library/Developer/Xcode

  

Clean Trash.
  

Re-install Xcode from APP Store

  

Swith xcode install dir to /Applications/Xcode.app  
sudo xcode-select --switch /Applications/Xcode.app

  

Install command line tools  
xcode-select --install
```

###  Could not find a valid Boot JDK.
错误信息包含
>configure: Found potential Boot JDK using /usr/libexec/java_home
configure: Potential Boot JDK found at /Library/Java/JavaVirtualMachines/jdk-17.0.1.jdk/Contents/Home is incorrect JDK version (java version "17.0.1" 2021-10-19 LTS Java(TM) SE Runtime Environment (build 17.0.1+12-LTS-39) Java HotSpot(TM) 64-Bit Server VM (build 17.0.1+12-LTS-39, mixed mode, sharing)); ignoring
configure: (Your Boot JDK version must be one of: 18 19 20)
configure: Found potential Boot JDK using /usr/libexec/java_home -v 18
configure: Potential Boot JDK found at /Library/Java/JavaVirtualMachines/jdk-17.0.1.jdk/Contents/Home is incorrect JDK version (java version "17.0.1" 2021-10-19 LTS Java(TM) SE Runtime Environment (build 17.0.1+12-LTS-39) Java HotSpot(TM) 64-Bit Server VM (build 17.0.1+12-LTS-39, mixed mode, sharing)); ignoring
configure: (Your Boot JDK version must be one of: 18 19 20)
configure: Found potential Boot JDK using /usr/libexec/java_home -v 19
configure: Potential Boot JDK found at /Library/Java/JavaVirtualMachines/jdk-17.0.1.jdk/Contents/Home is incorrect JDK version (java version "17.0.1" 2021-10-19 LTS Java(TM) SE Runtime Environment (build 17.0.1+12-LTS-39) Java HotSpot(TM) 64-Bit Server VM (build 17.0.1+12-LTS-39, mixed mode, sharing)); ignoring
configure: (Your Boot JDK version must be one of: 18 19 20)
configure: Found potential Boot JDK using /usr/libexec/java_home -v 20
configure: Potential Boot JDK found at /Library/Java/JavaVirtualMachines/jdk-17.0.1.jdk/Contents/Home is incorrect JDK version (java version "17.0.1" 2021-10-19 LTS Java(TM) SE Runtime Environment (build 17.0.1+12-LTS-39) Java HotSpot(TM) 64-Bit Server VM (build 17.0.1+12-LTS-39, mixed mode, sharing)); ignoring
configure: (Your Boot JDK version must be one of: 18 19 20)
checking for javac... /Users/carota/.jenv/shims/javac
checking for java... /Users/carota/.jenv/shims/java
configure: Found potential Boot JDK using well-known locations (in /Library/Java/JavaVirtualMachines/zulu-8.jdk)
configure: Potential Boot JDK found at /Library/Java/JavaVirtualMachines/zulu-8.jdk/Contents/Home is incorrect JDK version (openjdk version "1.8.0_322" OpenJDK Runtime Environment (Zulu 8.60.0.21-CA-macos-aarch64) (build 1.8.0_322-b06) OpenJDK 64-Bit Server VM (Zulu 8.60.0.21-CA-macos-aarch64) (build 25.322-b06, mixed mode)); ignoring
configure: (Your Boot JDK version must be one of: 18 19 20)
configure: Found potential Boot JDK using well-known locations (in /Library/Java/JavaVirtualMachines/zulu-11.jdk)
configure: Potential Boot JDK found at /Library/Java/JavaVirtualMachines/zulu-11.jdk/Contents/Home is incorrect JDK version (openjdk version "11.0.13" 2021-10-19 LTS OpenJDK Runtime Environment Zulu11.52+13-CA (build 11.0.13+8-LTS) OpenJDK 64-Bit Server VM Zulu11.52+13-CA (build 11.0.13+8-LTS, mixed mode)); ignoring
configure: (Your Boot JDK version must be one of: 18 19 20)
configure: Found potential Boot JDK using well-known locations (in /Library/Java/JavaVirtualMachines/openjdk-11.jdk)
configure: Potential Boot JDK found at /Library/Java/JavaVirtualMachines/openjdk-11.jdk/Contents/Home is incorrect JDK version (openjdk version "11.0.12" 2021-07-20 OpenJDK Runtime Environment Homebrew (build 11.0.12+0) OpenJDK 64-Bit Server VM Homebrew (build 11.0.12+0, mixed mode)); ignoring
configure: (Your Boot JDK version must be one of: 18 19 20)
configure: Found potential Boot JDK using well-known locations (in /Library/Java/JavaVirtualMachines/jdk-17.0.1.jdk)
configure: Potential Boot JDK found at /Library/Java/JavaVirtualMachines/jdk-17.0.1.jdk/Contents/Home is incorrect JDK version (java version "17.0.1" 2021-10-19 LTS Java(TM) SE Runtime Environment (build 17.0.1+12-LTS-39) Java HotSpot(TM) 64-Bit Server VM (build 17.0.1+12-LTS-39, mixed mode, sharing)); ignoring
configure: (Your Boot JDK version must be one of: 18 19 20)
configure: Could not find a valid Boot JDK. OpenJDK distributions are available at http://jdk.java.net/.
configure: This might be fixed by explicitly setting --with-boot-jdk
configure: error: Cannot continue
/Users/carota/Projects/Java/jdk/build/.configure-support/generated-configure.sh: line 84: 5: Bad file descriptor
configure exiting with result code 1

**错误原因**
因为编译JDK版本需要用到另一编译期可用的JDK.
笔者的JDK版本为JDK17,  而当时git下载下来的版本为最新的JDK20, JDK 17不能作为Boot JDK.

**解决方法**
到上文拉取的源码库下, 切换到其他版本的源码
笔者切换至JDK18的版本来进行编译
`git checkout tags/jdk-18+8`

### 'ptrauth.h' file not found
[参考来源](https://blog.swesonga.org/2022/01/12/exploring-the-hsdis-llvm-support-pr/)

`clang --version`
查看是否是apple 的clang版本, 因为我之前intel版本下的macos迁移过来,  且使用的是之前brew安装的llvm
**解决方法**
1. 使用brew uninstall来删除
2. clang --version 来确认是否是apple版本

### For target hotspot_variant-server_libjvm_objs_copy_bsd_aarch64.o: error: invalid integral value '16-DMAC_OS_X_VERSION_MIN_REQUIRED=110000' in '-mstack-alignment=16-DMAC_OS_X_VERSION_MIN_REQUIRED=110000

切换JDK18版本的具体小版本。jdk-18+22

### error: parameter 'SizeOfTag' set but not used [-Werror,-Wunused-but-set-parameter]

[OpenJDK-BUG跟踪](https://bugs.openjdk.org/browse/JDK-8283735)
切换JDK19版本来编译, 需要本地JDK为18版本