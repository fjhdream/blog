内容总结来源:
1. https://www.oracle.com/java/graalvm/what-is-graalvm/
2. https://www.graalvm.org/latest/reference-manual/native-image/
3. https://www.oracle.com/a/ocom/docs/graalvm-enterprise-white-paper.pdf
4. https://www.graalvm.org/latest/reference-manual/native-image/basics/

  

# 是个啥

GraalVM始于2011年，作为Oracle实验室的一个研究项目，旨在创建一个可以运行多种编程语言并具有高性能的运行时平台。

GraalVM项目的核心是高级优化GraalVM编译器，它被用作Java虚拟机（JVM）的即时（JIT）编译器，或者被GraalVM原生映像功能提前编译Java字节码为原生机器代码。GraalVM的Truffle语言实现框架与GraalVM编译器协同工作，在JVM上运行JavaScript、Python、Ruby和其他支持的语言，性能突出。。

  

1. GraalVM包括一个先进的即时编译器（Just-In-Time, **JIT**），名为GraalVM JI编译器，JVM使用GraalVM JI编译器在应用程序运行时从Java字节码创建特定于平台的机器代码。
2. 还有一个本机映像实用程序(**Native-Image**)可以提前编译Java字节码（**AOT**），并为一些几乎立即启动并使用很少内存资源的应用程序生成本机可执行文件。
3. GraalVM还通过**Truffle**语言实现框架支持多语言互操作性。Truffle使用支持的语言编写的程序能够使用多语言库。例如，JavaScript程序可以调用Ruby方法并共享值，而无需制作副本。在JVM上运行时，Truffle与GraalVM编译器协作，将支持的语言编译为本机机器代码，以获得最佳性能，就像Java一样。(https://www.graalvm.org/latest/graalvm-as-a-platform/language-implementation-framework/)

  

# 特性

1. 更少的资源占用
> Native executables use only a fraction of memory and CPU resources required by a JVM, which improves utilization and reduces costs.

2. 更安全
> Native executables contain only the classes, methods, and fields that your application needs, which reduces attack surface area.

3. 启动更快
> Native executables compiled ahead of time start up instantly and require no warmup to run at peak performance.

4. 打包体积更小
> Native executables are small and offer a range of linking options that make them easy to deploy in minimal container images.

5. 框架支持良好
> Popular frameworks such as Spring Boot, Micronaut, Helidon, and Quarkus provide first-class support for GraalVM.

6. 云平台支持
> SDKs from leading cloud platforms such as AWS, Microsoft Azure, GCP, and Oracle Cloud Infrastructure integrate and support GraalVM.

# 用处

1. **Boost performance and extend existing Java applications**
	JI 编译器利用高级优化器提高了峰值吞吐量。它还通过最小化对象分配来优化内存消耗，以减少执行垃圾回收机制所花费的时间。GraalVM在JIT模式下运行可以提升性能高达50%。这可以更快地释放内存，降低IT成本。

2. **Build cloud native applications**
      Oracle GraalVM的本机映像实用程序将字节码Java应用程序提前编译成机器二进制文件。与在JVM上运行相比，本机可执行文件的启动速度快了近100倍，消耗的内存少了5倍。
      HelloWorld运行对比
    ![](https://fyze31atzb.feishu.cn/space/api/box/stream/download/asynccode/?code=ZjQ0NWYxMmQyODRkYmUxMGY1ZTU4OTM5YWY5ZmFjZDVfbTc4alpGaDRURXQzUDZRQXRES01JTEVsQWxRTVFraERfVG9rZW46RnVIa2JvQ0Vwb3drRGh4blRSbmM4UHVCbjhjXzE3MDAxMjY2NDU6MTcwMDEzMDI0NV9WNA)
    
2. **Develop multilanguage programs and improve productivity**

可以在JVM上以优异的性能运行Java和其他语言，如JavaScript。Oracle NetSuite的SuiteScript是一种扩展语言，供最终用户自定义在NetSuite上运行的应用程序。Oracle GraalVM使NetSuite嵌入了一个基于Truffle的JavaScript运行时，可以将JavaScript编译成比在JVM上运行的SuiteScript快4倍的机器代码。

# Native executable? 二进制文件直接执行？

是谁把Java变成一个二进制可执行文件的？ **Native-Image**

怎么做到的？

> native-image processes your application classes and [other metadata](https://www.graalvm.org/latest/reference-manual/native-image/metadata/) to create a binary for a specific operating system and architecture.
> 
> First, the `native-image` tool performs static analysis of your code to determine the classes and methods that are reachable when your application runs. Second, it compiles classes, methods, and resources into a binary. This entire process is called build time to clearly distinguish it from the compilation of Java source code to bytecode.

> native-image工具会处理你的应用程序类和其他元数据，为特定的操作系统和架构创建一个二进制文件。
> 
> 首先，native-image工具对你的代码进行静态分析，确定在应用程序运行时可达到的类和方法。其次，它将类、方法和资源编译成一个二进制文件。整个过程被称为构建时间（build time），以明确区分它与Java源代码编译成字节码的过程。

# 实践一下

## HelloWord

```Java
public class HelloWorld {
    static class Greeter {
        static {
            System.out.println("Greeter is getting ready!");
        }

        public static void greet() {
            System.out.println("Hello, World!");
        }
    }
    public static void main(String[] args) {
        Greeter.greet();
    }
}
```

```Shell
 javac HelloWorld.java
 java Helloworld
 
 
 native-image HelloWorld
 ./helloWorld
 
 # 告诉natvie-image 构建的时候初始化静态类
 native-image HelloWorld --initialize-at-build-time=HelloWorld\$Greeter
  ./helloWorld
```

### **Build Time(构建) vs Run Time(运行)**

在图像构建过程中，本地图像可能会执行用户代码。这段代码可能会产生副作用，比如向类的静态字段写入一个值。我们称这段代码在构建时执行。由此代码写入的静态字段的值保存在图像堆中。运行时是指二进制代码在执行时的代码和状态。

在Java中，当一个类首次被使用时，它会被初始化。在构建时使用的每个Java类都被称为**构建时初始化(build-time initialized)**。请注意，仅仅加载一个类并不一定会初始化它。构建时初始化类的静态类初始化器在运行镜像构建的JVM上执行(The static class initializer of build-time initialized classes executes on the JVM running the image build)。

如果一个类在构建时被初始化，它的静态字段会保存在生成的二进制文件中。在运行时，首次使用这样的类不会触发类初始化。

## 复杂点的例子

```Java
class Example {
    private static final String message;
    
    static {
        message = System.getProperty("message");
    }

    public static void main(String[] args) {
        System.out.println("Hello, World! My message is: " + message);
    }
}
```

```Shell
javac Example.java
java -Dmessage=hi Example
java -Dmessage=hello Example 
java Example

native-image Example --initialize-at-build-time=Example -Dmessage=native
./example 
./example -Dmessage=aNewMessage

native-image
./example -Dmessage=aNewMessage
./example -Dmessage=Hello
```

### 本地图像堆(Native Image Heap / image heap)
包含以下三部分:

1. Objects created during the image build that are reachable from application code.
> 从应用程序代码中可访问的在图像构建过程中创建的对象。

2. `java.lang.Class` objects of classes used in the native image.
> 在本地镜像中使用的类的对象。

3. Object constants [embedded in method code](https://www.graalvm.org/latest/reference-manual/native-image/metadata/#computing-metadata-in-code).
> 方法代码中嵌入的对象常量

```Java
 class ReflectiveAccess {
     public Class<Foo> fetchFoo() throws ClassNotFoundException {
         return Class.forName("Foo");
     }
 }
```

### 静态分析(Static Analysis)

静态分析是一种确定应用程序使用哪些程序元素（类、方法和字段）的过程。这些元素也被称为**可达代码(reachable code)**。分析本身有两个部分：

1. 扫描方法的字节码以确定从中可达到的其他元素。
2. 扫描本地镜像堆中的根对象（即静态字段），以确定哪些类可以从它们访问。它从应用程序的入口点开始（即 `main` 方法）。新发现的元素会被迭代地扫描，直到进一步的扫描不再改变元素的可达性。

最终图像中只包含可达元素。一旦构建了本地图像，就不能在运行时添加新元素，例如通过类加载。我们将这个约束称为**封闭世界假设(closed-world assumption)**。

## Jar包编译来一下

目录

```Shell
  | src
  |   --com/
  |      -- example
  |          -- App.java
```

代码

```Java
  package com.example;

  public class App {

      public static void main(String[] args) {
          String str = "Native Image is awesome";
          String reversed = reverseString(str);
          System.out.println("The reversed string is: " + reversed);
      }

      public static String reverseString(String str) {
          if (str.isEmpty())
              return str;
          return reverseString(str.substring(1)) + str.charAt(0);
      }
  }
```

编译命令

```Bash
1.  javac -d build src/com/example/App.java
2.  jar --create --file App.jar --main-class com.example.App -C build .
3.  native-image -jar App.jar
```

Tips：

The default behavior of `native-image` is aligned with the `java` command which means you can pass the `-jar`, `-cp`, `-m` options to build with Native Image as you would normally do with `java`. For example, `java -jar App.jar someArgument` becomes `native-image -jar App.jar` and `./App someArgument`.

### 构建内容输出(Native Image Build Output)

1. Build Stages 构建阶段
2. Resource Usage Statistics 资源使用统计
3. Machine-Readable Build Output 机器可读的构建输出

更多详细情查看:  https://www.graalvm.org/latest/reference-manual/native-image/overview/BuildOutput/

``` Java
========================================================================================================================
GraalVM Native Image: Generating 'App' (executable)...
========================================================================================================================
[1/8] Initializing...                                                                                    (5.9s @ 0.11GB)
 Java version: 21+35, vendor version: Oracle GraalVM 21+35.1
 Graal compiler: optimization level: 2, target machine: armv8-a, PGO: off
 C compiler: cc (apple, arm64, 15.0.0)
 Garbage collector: Serial GC (max heap size: 80% of RAM)
 1 user-specific feature(s):
 - com.oracle.svm.thirdparty.gson.GsonFeature
------------------------------------------------------------------------------------------------------------------------
Build resources:
 - 24.18GB of memory (75.6% of 32.00GB system memory, determined at start)
 - 10 thread(s) (100.0% of 10 available processor(s), determined at start)
[2/8] Performing analysis...  [******]                                                                   (3.5s @ 0.22GB)
    2,085 reachable types   (60.9% of    3,421 total)
    2,000 reachable fields  (45.9% of    4,361 total)
    9,590 reachable methods (38.4% of   24,950 total)
      764 types,   109 fields, and   474 methods registered for reflection
       49 types,    33 fields, and    48 methods registered for JNI access
        4 native libraries: -framework Foundation, dl, pthread, z
[3/8] Building universe...                                                                               (0.7s @ 0.28GB)
[4/8] Parsing methods...      [*]                                                                        (0.5s @ 0.28GB)
[5/8] Inlining methods...     [***]                                                                      (0.4s @ 0.26GB)
[6/8] Compiling methods...    [***]                                                                      (7.9s @ 0.30GB)
[7/8] Layouting methods...    [*]                                                                        (0.7s @ 0.32GB)
[8/8] Creating image...       [*]                                                                        (1.2s @ 0.24GB)
   3.21MB (44.81%) for code area:     4,517 compilation units
   3.75MB (52.33%) for image heap:   57,014 objects and 71 resources
 209.98kB ( 2.86%) for other data
   7.17MB in total
------------------------------------------------------------------------------------------------------------------------
Top 10 origins of code area:                                Top 10 object types in image heap:
   1.71MB java.base                                          839.82kB byte[] for code metadata
   1.23MB svm.jar (Native Image)                             718.58kB byte[] for java.lang.String
  82.86kB com.oracle.svm.svm_enterprise                      438.15kB heap alignment
  40.44kB jdk.proxy3                                         382.41kB java.lang.String
  38.48kB jdk.proxy1                                         330.89kB java.lang.Class
  25.43kB org.graalvm.nativeimage.base                       154.22kB java.util.HashMap$Node
  22.78kB org.graalvm.collections                            114.01kB char[]
  13.92kB jdk.internal.vm.ci                                  99.18kB byte[] for reflection metadata
  13.61kB jdk.internal.vm.compiler                            91.62kB java.lang.Object[]
  11.51kB jdk.proxy2                                          81.45kB com.oracle.svm.core.hub.DynamicHubCompanion
   1.61kB for 2 more packages                                589.69kB for 557 more object types
                              Use '-H:+BuildReport' to create a report with more details.
------------------------------------------------------------------------------------------------------------------------
Security report:
 - Binary does not include Java deserialization.
 - Use '--enable-sbom' to embed a Software Bill of Materials (SBOM) in the binary.
------------------------------------------------------------------------------------------------------------------------
Recommendations:
 PGO:  Use Profile-Guided Optimizations ('--pgo') for improved throughput.
 INIT: Adopt '-H:+StrictImageHeap' to prepare for the next GraalVM release.
 HEAP: Set max heap for improved and more predictable memory usage.
 CPU:  Enable more CPU features with '-march=native' for improved performance.
 QBM:  Use the quick build mode ('-Ob') to speed up builds during development.
------------------------------------------------------------------------------------------------------------------------
                        1.0s (4.8% of total time) in 251 GCs | Peak RSS: 0.82GB | CPU load: 4.95
------------------------------------------------------------------------------------------------------------------------
Produced artifacts:
 /Users/carota/Projects/Java/HelloNativeJar/App (executable)
========================================================================================================================
Finished generating 'App' in 21.1s.
```

> Tips:
>  [RSS](https://en.wikipedia.org/wiki/Resident_set_size): 在计算机中，驻留集大小（RSS）是进程占用的主存储器（RAM）中的内存部分（以千字节为单位）。其余占用的内存存在于交换空间或文件系统中，这可能是因为部分占用的内存被分页出去，或者因为部分可执行文件未加载。



## 项目那么多第三方库怎么办？

### Reachability Metadata 可达性元数据

构建分析可以确定一些动态类加载的情况，但它并不能始终完全预测Java Native Interface (JNI)、Java Reflection、Dynamic Proxy对象或类路径资源的所有用法。为了处理Java的这些动态特性，您需要向分析提供使用Reflection、Proxy等功能的类的详细信息，或者要动态加载哪些类。


为了确保将这些元素包含到本机二进制文件中，您应该向 `native-image` 构建器提供可达性元数据（在后文中称为元数据）。向构建器提供可达性元数据还可以确保与第三方库在运行时的无缝兼容性。

**两种方式:**
1. 通过在本地二进制文件构建时计算代码中的元数据，并将所需元素存储到本地二进制文件的初始堆中。
2. 通过提供存储在 `META-INF/native-image/<group.id>/<artifact.id>` 项目目录中的JSON文件

  

**命名格式:**
每个需要元数据的动态Java功能都有一个对应的JSON文件，文件名为 `<feature>-config.json` 。JSON文件由条目组成，告诉Native Image要包含的元素。例如，Java反射元数据在 `reflect-config.json` 中指定

**MetadataTypes:**
1. Java Reflection
2. JNI
3. Resources and Resource Bundles
4. Dynamic JDK Proxies
5. Serialization
6. Predefined Classes

详细与各个元数据类型交互细节见: https://www.graalvm.org/latest/reference-manual/native-image/dynamic-features/

  
### Tracing Agent
自己写太麻烦? GraalVM提供个[Tracing Agent](https://www.graalvm.org/latest/reference-manual/native-image/metadata/AutomaticMetadataCollection/)

1. -agentlib:native-image-agent=....
2. 通过进程环境注入代理

```Java
export JAVA_TOOL_OPTIONS="-agentlib:native-image-agent=config-output-dir=/path/to/config-output-dir-{pid}-{datetime}/"
```

To learn more about metadata, ways to provide it, and supported metadata types, see [Reachability Metadata](https://www.graalvm.org/latest/reference-manual/native-image/metadata/). To automatically collect metadata for your application, see [Automatic Collection of Metadata](https://www.graalvm.org/latest/reference-manual/native-image/metadata/AutomaticMetadataCollection/).

## 我要开箱即用！

拥抱SpringBoot3 https://docs.spring.io/spring-boot/docs/current/reference/html/native-image.html

### 示例 简单的CRUD SpringBoot3 程序

**上Git链接**

https://github.com/fjhdream/HelloSpringNative

``` java
package dream.js.hellospringnative;  
  
import dream.js.hellospringnative.entity.Person;  
import dream.js.hellospringnative.repository.PersonRepository;  
import org.springframework.beans.factory.annotation.Autowired;  
import org.springframework.web.bind.annotation.GetMapping;  
import org.springframework.web.bind.annotation.RestController;  
  
@RestController  
public class CommonController {  
  
    private PersonRepository personRepository;  
  
    public CommonController(@Autowired PersonRepository personRepository) {  
        this.personRepository = personRepository;  
    }  
  
    @GetMapping("/persons")  
    public Iterable<Person> persons() {  
        return personRepository.findAll();  
    }  
  
    @GetMapping("/hello")  
    public String hello() {  
        return "world";  
    }  
}
```
# 数据对比

1. JIT vs NativeImage

![](https://fyze31atzb.feishu.cn/space/api/box/stream/download/asynccode/?code=YTE0YTQ1ZjRjZTUwN2MzYzQ0MThkMzRlODAyZGM2ZmRfR3UzWnI2WUpia1hpRkFSR3VqTmx0MXBDUzRvUHVlT21fVG9rZW46VVozRWJ3TlZrb1VvVW14elhRSmNzUkp5blNiXzE3MDAxMjY2NDU6MTcwMDEzMDI0NV9WNA)

Performance of Spring Petclinic with Oracle GraalVM Native Image and GraalVM CE with C2 JIT. The benchmarking experiment ran the latest [Spring Petclinic](https://github.com/spring-projects/spring-petclinic) on Oracle X5–2 servers (Intel Xeon E5–2699 v3), restricting the workload to 16 CPUs and setting a maximum heap size of 1GB.

2. 性能基准对比

![](https://fyze31atzb.feishu.cn/space/api/box/stream/download/asynccode/?code=OTg1ODEyMmFkYTg3OWJjMGQ5YTc2MjQ3MWVmMzdjNDVfcmFVeHIzVzZvcnlJZ1hpNXZiMENoWGtXNXF5Z3RmSXpfVG9rZW46T0lIUmJuWFNvb0hDZWp4MkdHcWNzR0MwbkplXzE3MDAxMjY2NDU6MTcwMDEzMDI0NV9WNA)

Performance of Spring Petclinic with Oracle GraalVM Native Image and GraalVM CE with C2 JIT. The benchmarking experiment ran the latest [Spring Petclinic](https://github.com/spring-projects/spring-petclinic) on Oracle X5–2 servers (Intel Xeon E5–2699 v3), restricting the workload to 16 CPUs and setting a maximum heap size of 512MB.

3. 吞吐量对比

![](https://fyze31atzb.feishu.cn/space/api/box/stream/download/asynccode/?code=MTI5YTE4MTFiMjVlNjEzMGE3MjhlZDY4NDFhZmIxMGJfT1hRY1dtbXMwZmZGSVlWSWhvWG43QVV2V21yazlTa3FfVG9rZW46WU9TS2JIcHlob3d4S1d4VDNZS2M0ZFNvblpkXzE3MDAxMjY2NDU6MTcwMDEzMDI0NV9WNA)