## Maven 是什么

Maven 是一个基于项目对象模型（POM）的项目管理和构建自动化工具，主要用于 Java 项目。Maven 通过描述项目的构建过程、依赖管理和项目信息来简化开发过程。以下是 Maven 的一些主要功能和特点：

1. **项目构建**：Maven 使用 POM 文件（pom.xml）来管理项目的构建过程，包括编译、测试、打包和部署。
2. **依赖管理**：Maven 可以自动处理项目的依赖项，下载所需的库和插件，确保所有依赖项的一致性和兼容性。
3. **生命周期管理**：Maven 定义了一系列标准的生命周期阶段，如验证、编译、测试、打包、集成测试、验证和部署。每个阶段包含一组标准的任务，可以按顺序执行。
4. **插件机制**：Maven 通过插件来扩展其功能。不同的插件可以执行特定的任务，如编译代码、运行测试、生成报告等。
5. **仓库管理**：Maven 使用本地和远程仓库来存储项目的依赖项。中央仓库（Maven Central）是最常用的远程仓库。
6. **项目模板**：Maven 提供了一组标准的项目模板（Archetype），使开发者可以快速创建新项目的基本结构。

## Maven是如何查询下载依赖的

Maven都是从Repository拉取数据的, Mirror只是某个Repository的拷贝.

``` xml
<settings xmlns="http://maven.apache.org/SETTINGS/1.0.0"
          xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
          xsi:schemaLocation="http://maven.apache.org/SETTINGS/1.0.0
                              http://maven.apache.org/xsd/settings-1.0.0.xsd">
  
  <localRepository>/path/to/local/repo</localRepository>

  <mirrors>
    <mirror>
      <id>example-mirror</id>
      <mirrorOf>central</mirrorOf>
      <url>http://example.com/maven2</url>
    </mirror>
  </mirrors>

  <profiles>
    <profile>
      <id>default</id>
      <repositories>
        <repository>
          <id>central</id>
          <url>https://repo.maven.apache.org/maven2</url>
          <releases>
            <enabled>true</enabled>
          </releases>
          <snapshots>
            <enabled>false</enabled>
          </snapshots>
        </repository>
        <repository>
          <id>company-repo</id>
          <url>http://repo.company.com/maven2</url>
          <releases>
            <enabled>true</enabled>
          </releases>
          <snapshots>
            <enabled>true</enabled>
          </snapshots>
        </repository>
      </repositories>
    </profile>
  </profiles>

  <activeProfiles>
    <activeProfile>default</activeProfile>
  </activeProfiles>

</settings>

```

![Maven 2024-07-09_20.22.40.excalidraw](https://picbed.fjhdream.cn/202407100936502.svg)  
%%[[../../../_excalidraw/Maven 2024-07-09_20.22.40.excalidraw.md|🖋 Edit in Excalidraw]]%%

## Maven Profile示例

### `pom.xml` 中的 `profile`

在 `pom.xml` 文件中，`profile` 用于定义项目的不同配置。例如，你可以为开发环境、测试环境和生产环境定义不同的配置。一个典型的 `profile` 配置如下

``` xml
<profiles>
    <profile>
        <id>development</id>
        <properties>
            <environment>dev</environment>
        </properties>
        <dependencies>
            <!-- 开发环境的依赖 -->
        </dependencies>
        <build>
            <plugins>
                <!-- 开发环境的插件配置 -->
            </plugins>
        </build>
    </profile>
    
    <profile>
        <id>production</id>
        <properties>
            <environment>prod</environment>
        </properties>
        <dependencies>
            <!-- 生产环境的依赖 -->
        </dependencies>
        <build>
            <plugins>
                <!-- 生产环境的插件配置 -->
            </plugins>
        </build>
    </profile>
</profiles>
```

### `settings.xml` 中的 `profile`

在 `settings.xml` 文件中，你可以定义用户或机器特定的配置文件。这通常用于定义一些全局的、与项目无关的配置。一个典型的 `settings.xml` 配置如下：

``` xml
<profiles>
    <profile>
        <id>development</id>
        <properties>
            <maven.repo.local>/path/to/local/repo</maven.repo.local>
        </properties>
    </profile>
    
    <profile>
        <id>production</id>
        <properties>
            <maven.repo.local>/another/path/to/local/repo</maven.repo.local>
        </properties>
    </profile>
</profiles>
<activeProfiles>
    <activeProfile>development</activeProfile>
</activeProfiles>
```

### `pom.xml` 和 `settings.xml` 中 `profile` 的匹配使用

当你在 `pom.xml` 和 `settings.xml` 中定义了同名的 `profile` 时，它们可以共同作用，组合使用。例如，你可以在 `settings.xml` 中定义某些全局的属性，而在 `pom.xml` 中定义项目特定的依赖和插件配置。

匹配使用的方法如下：

1. **定义 `profile`**：在 `pom.xml` 和 `settings.xml` 中都定义相同的 `profile` ID。
2. **激活 `profile`**：可以在 `settings.xml` 中使用 `<activeProfiles>` 节点激活某个 `profile`，也可以在命令行使用 `-P` 参数激活。例如，使用命令 `mvn clean install -P production` 激活 `production` 配置。

### 示例

假设你在 `settings.xml` 中定义了一个 `development` 的 `profile`，并且激活了它：

``` xml
<activeProfiles>     
	<activeProfile>development</activeProfile> 
</activeProfiles>
```

在 `pom.xml` 中定义了相应的 `profile`：

``` xml
<profiles>
    <profile>
        <id>development</id>
        <properties>
            <environment>dev</environment>
        </properties>
        <dependencies>
            <dependency>
                <groupId>com.example</groupId>
                <artifactId>dev-only-dependency</artifactId>
                <version>1.0.0</version>
            </dependency>
        </dependencies>
    </profile>
</profiles>

```

在这种情况下，当你运行 Maven 构建时，`development` profile 将被激活，`pom.xml` 和 `settings.xml` 中的配置将共同作用。