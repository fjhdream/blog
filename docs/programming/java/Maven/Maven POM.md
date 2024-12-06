POM 代表“项目对象模型”。它是 Maven 项目的 XML 表示，保存在名为 `pom.xml` 的文件中。

当与 Maven 的人在一起时，谈论一个项目是在哲学意义上谈论，超越了仅仅包含代码的文件集合。一个项目包含配置文件，以及参与其中的开发人员及其扮演的角色，缺陷跟踪系统，组织和许可证，项目所在位置的 URL，项目的依赖关系，以及所有其他起作用的小部分，以赋予代码生命。这是一个关于项目所有事项的一站式服务。实际上，在 Maven 世界中，一个项目根本不需要包含任何代码，仅仅一个 `pom.xml` 。

## 快速预览

这是在 POM 项目元素下直接列出的元素清单。请注意， `modelVersion` 包含 4.0.0。这是目前唯一支持的 POM 版本，且始终是必需的。

``` xml
<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"

xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">

	<modelVersion>4.0.0</modelVersion>
	
	<!-- The Basics -->
	<groupId>...</groupId>
	<artifactId>...</artifactId>
	<version>...</version>
	<packaging>...</packaging>
	<dependencies>...</dependencies>
	<parent>...</parent>
	<dependencyManagement>...</dependencyManagement>
	<modules>...</modules>
	<properties>...</properties>
	
	<!-- Build Settings -->
	<build>...</build>
	<reporting>...</reporting>
	
	<!-- More Project Information -->
	<name>...</name>
	<description>...</description>
	<url>...</url>
	<inceptionYear>...</inceptionYear>
	<licenses>...</licenses>
	<organization>...</organization>
	<developers>...</developers>
	<contributors>...</contributors>
	
	<!-- Environment Settings -->
	<issueManagement>...</issueManagement>
	<ciManagement>...</ciManagement>
	<mailingLists>...</mailingLists>
	<scm>...</scm>
	<prerequisites>...</prerequisites>
	<repositories>...</repositories>
	<pluginRepositories>...</pluginRepositories>
	<distributionManagement>...</distributionManagement>
	<profiles>...</profiles>
</project>
```

## Basic

POM 包含有关项目的所有必要信息，以及在构建过程中要使用的插件配置。它是“谁”、“什么”和“在哪里”的声明性表现，而构建生命周期是“何时”和“如何”。这并不是说 POM 不能影响生命周期的流程 - 它可以。

``` xml
<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">

	<modelVersion>4.0.0</modelVersion>
	<groupId>org.codehaus.mojo</groupId>
	<artifactId>my-project</artifactId>
	<version>1.0</version>

</project>
```

## Maven Coordinates

上面定义的 POM 是 Maven 允许的最低要求。 `groupId:artifactId:version` 都是必填字段（尽管如果它们是从父级继承而来，则 groupId 和 version 不需要明确定义 - 更多关于继承的内容稍后）。这三个字段就像一个地址和时间戳的组合。这标记了存储库中的特定位置，就像 Maven 项目的坐标系

- **groupId**: 这通常在组织或项目中是唯一的。
- **artifactId**: artifactId 通常是项目所知的名称。
- **version**: `groupId:artifactId` 表示一个项目，但它们无法区分我们正在谈论的项目的哪个具体版本。  
上面提供的三个元素指向一个项目的特定版本，让 Maven 知道我们正在处理谁，以及在软件生命周期的哪个阶段我们想要它们。

## Packaging 

现在我们有了 `groupId:artifactId:version` 的地址结构，还有一个标准标签可以让我们真正完整：那就是项目的打包。

``` xml
<packaging>war</packaging>
```

当没有声明包装时，Maven 假定包装是默认值: `jar` 。

## POM Relationships

Maven 的一个强大特性是它对项目关系的处理：这包括依赖关系（以及传递依赖关系）、继承和聚合（多模块项目）。

Maven 通过一个共同的本地存储库解决了这两个问题，可以正确地链接项目、版本和所有内容。

### Dependencies

``` xml
<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">

<dependencies>
	<dependency>
		<groupId>junit</groupId>
		<artifactId>junit</artifactId>
		<version>4.12</version>
		<type>jar</type>
		<scope>test</scope>
		<optional>true</optional>
	</dependency>
</dependencies>

</project>
```

- **scope**: 范围：  
    此元素指的是当前任务的类路径（编译和运行时，测试等）以及如何限制依赖关系的传递性。有五种可用的范围：
    - **compile**  
        编译 - 这是默认范围，如果未指定范围，则使用。编译依赖项在所有类路径中都可用。此外，这些依赖项会传播到依赖项目。
    - **provided**  
        提供 - 这很像编译，但表示您期望 JDK 或容器在运行时提供它。 它仅在编译和测试类路径上可用，并且不是传递的。
    - **runtime**   
        运行时 - 这个范围表示依赖项不是编译所必需的，但是用于执行。它在运行时和测试类路径中，但不在编译类路径中。
    - **test**  
        测试 - 这个范围表示该依赖项对应用程序的正常使用并不是必需的，仅在测试编译和执行阶段可用。它不是传递性的。
    - **system**  
        系统 - 这个范围类似于 `provided` ，不同之处在于您必须明确提供包含它的 JAR 文件。该构件始终可用，不会在存储库中查找。
- **optional**: `optional` 让其他项目知道，当您使用此项目时，您不需要此依赖项才能正常工作。 例如，假设项目 `A` 依赖于项目 `B` 来编译一部分在运行时可能不会被使用的代码，那么我们可能不需要项目 `B` 。因此，如果项目 `X` 将项目 `A` 添加为自己的依赖项，那么 Maven 根本不需要安装项目 `B` 。在符号上，如果 `=>` 代表必需依赖项， `-->` 代表可选项，尽管在构建 A 时可能是 `A=>B` 的情况，在构建 `X` 时可能是 `X=>A-->B` 的情况。

### Exclusions

有时限制依赖项的传递依赖关系是有用的。 依赖项可能具有错误指定的范围，或者与项目中的其他依赖项冲突的依赖项。 排除告诉 Maven 不要在类路径中包含指定的工件，即使它是此项目依赖项的一个或多个的依赖项（传递依赖项）。

例如， `maven-embedder` 依赖于 `maven-core` 。 假设您想依赖于 maven-embedder，但不想在类路径中包含 maven-core 或其依赖项。 然后将 `maven-core` 添加为 `exclusion` ，在声明对 maven-embedder 依赖项的元素中：

``` xml
<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">
  <dependencies>
    <dependency>
      <groupId>org.apache.maven</groupId>
      <artifactId>maven-embedder</artifactId>
      <version>3.9.8</version>
      <exclusions>
        <exclusion>
          <groupId>org.apache.maven</groupId>
          <artifactId>maven-core</artifactId>
        </exclusion>
      </exclusions>
    </dependency>
  </dependencies>
</project>
```

**这只是从这个依赖项中删除了对 maven-core 的路径**。如果 maven-core 在 POM 的其他位置作为直接或传递依赖项出现，它仍然可以添加到类路径中。

也可以通过通配符排除依赖项

``` xml
<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">
  ...
  <dependencies>
    <dependency>
      <groupId>org.apache.maven</groupId>
      <artifactId>maven-embedder</artifactId>
      <version>3.8.6</version>
      <exclusions>
        <exclusion>
          <groupId>*</groupId>
          <artifactId>*</artifactId>
        </exclusion>
      </exclusions>
    </dependency>
    ...
  </dependencies>
  ...
</project>
```

### Inheritance 继承

Maven 带来的一个强大功能是项目继承的概念

`packaging` 类型需要对父级和聚合（多模块）项目进行 `pom` 。这些类型定义了绑定到一组生命周期阶段的目标。例如，如果打包是 `jar` ，那么 `package` 阶段将执行 `jar:jar` 目标。现在我们可以向父 POM 添加值，这些值将被其子项目继承。大多数来自父 POM 的元素都会被其子项目继承，包括：

- groupId 组 ID
- version 版本
- description 描述
- url
- inceptionYear 创立年份
- organization 组织
- licenses 许可证
- developers 开发者
- contributors 贡献者
- mailingLists 邮寄名单
- scm
- issueManagement
- ciManagement ci 管理
- properties 属性
- dependencyManagement 依赖管理
- dependencies 依赖
- repositories 存储库
- pluginRepositories 插件仓库
- build  建造
    - plugin executions with matching ids  
        具有匹配 ID 的插件执行
    - plugin configuration 插件配置
    - etc. 等等。
- reporting 报告

值得注意的 `not` 继承的元素包括：

- artifactId
- name 名称
- prerequisites 先决条件
- profiles (but the effects of active profiles from parent POMs are) 配置文件（但是来自父 POM 的活动配置文件的是）

``` xml
<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">
  <modelVersion>4.0.0</modelVersion>
 
  <parent>
    <groupId>org.codehaus.mojo</groupId>
    <artifactId>my-parent</artifactId>
    <version>2.0</version>
    <relativePath>../my-parent</relativePath>
  </parent>
 
  <artifactId>my-project</artifactId>
</project>
```

### Aggregation (Multi-Module)

``` xml
<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">
  <modelVersion>4.0.0</modelVersion>
 
  <groupId>org.codehaus.mojo</groupId>
  <artifactId>my-parent</artifactId>
  <version>2.0</version>
  <packaging>pom</packaging>
 
  <modules>
    <module>my-project</module>
    <module>another-project</module>
    <module>third-project/pom-example.xml</module>
  </modules>
</project>
```

您在列出模块时无需考虑模块之间的依赖关系；即 POM 给出的模块顺序并不重要。Maven 将对模块进行拓扑排序，以确保依赖关系始终在依赖模块之前构建。

## Properties 属性

属性是理解 POM 基础知识的最后一个必要部分。Maven 属性是值占位符，就像 Ant 中的属性一样。它们的值可以通过在 POM 中的任何地方使用符号 `${X}` 来访问，其中 `X` 是属性。或者它们可以被插件用作默认值，例如：

``` xml
<project>
  ...
  <properties>
    <maven.compiler.source>1.7</maven.compiler.source>
    <maven.compiler.target>1.7</maven.compiler.target>
    <!-- Following project.-properties are reserved for Maven in will become elements in a future POM definition. -->
    <!-- Don't start your own properties properties with project. -->
    <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding> 
    <project.reporting.outputEncoding>UTF-8</project.reporting.outputEncoding>
  </properties>
  ...
</project>
```

1. `env.X` ：在变量前加上"env."将返回 shell 的环境变量。例如， `${env.PATH}` 包含了 PATH 环境变量。  
    注意：虽然在 Windows 上环境变量本身是不区分大小写的，但属性的查找是区分大小写的。换句话说，虽然 Windows shell 对 `%PATH%` 和 `%Path%` 返回相同的值，但 Maven 区分 `${env.PATH}` 和 `${env.Path}` 。为了可靠性起见，环境变量的名称被规范化为全大写。
2. `project.x` ：POM 中的点（.）标记路径将包含相应元素的值。例如： `<project><version>1.0</version></project>` 可通过 `${project.version}` 访问。
3. `settings.x` ：在 `settings.xml` 中以点（.）标记的路径将包含相应元素的值。例如： `<settings><offline>false</offline></settings>` 可通过 `${settings.offline}` 访问。
4. Java 系统属性：通过 `java.lang.System.getProperties()` 访问的所有属性都可以作为 POM 属性使用，例如 `${java.home}` 。
5. `x` ：在 POM 中的 `<properties />` 元素中设置。 `<properties><someVar>value</someVar></properties>` 的值可以用作 `${someVar}`

## Build Settings

### Build

根据 POM 4.0.0 XSD， `build` 元素在概念上分为两部分：有一个 `BaseBuild` 类型，其中包含对两个 `build` 元素共同的元素集；还有一个 `Build` 类型，其中包含 `BaseBuild` 集合以及更多顶级定义的元素。让我们从分析这两者之间的共同元素开始。

#### BaseBuild Element

``` xml
<build>
  <defaultGoal>install</defaultGoal>
  <directory>${project.basedir}/target</directory>
  <finalName>${artifactId}-${version}</finalName>
  <filters>
    <filter>filters/filter1.properties</filter>
  </filters>
  ...
</build>
```

- **defaultGoal**: 默认目标：如果没有指定目标或阶段，则执行的默认目标或阶段。如果给定了目标，则应该像在命令行中那样定义（例如 `jar:jar` ）。如果定义了阶段（例如 install），也是如此。
- **directory: 目录：这是构建将转储其文件的目录，或者用 Maven 术语说，构建的目标。它默认为 `${project.basedir}/target` 
- **finalName**: 这是bundled项目最终构建时的名称（不包括文件扩展名，例如： `my-project-1.0.jar` ）。默认为 `${artifactId}-${version}` 。
- **filter**: 过滤器：定义包含一组适用于接受其设置的资源的属性列表的 `*.properties` 文件
	- 换句话说，在构建时，过滤器文件中定义的“ `name=value` ”对会替换资源中的 `${name}` 字符串。上面的示例定义了 `filters/` 目录下的 `filter1.properties` 文件。
	- Maven 的默认过滤器目录是 `${project.basedir}/src/main/filters/`

#### Resources

`build` 元素的另一个特点是指定项目中资源存在的位置

例如，Plexus 项目需要一个 `configuration.xml` 文件（指定组件配置到容器）存放在 `META-INF/plexus` 目录中。虽然我们可以将这个文件放在 `src/main/resources/META-INF/plexus` 中，但我们希望给 Plexus 分配一个独立的目录 `src/main/plexus` 。为了让 JAR 插件正确捆绑资源，您需要指定类似以下的资源：

``` xml
<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">
  <build>
    ...
    <resources>
      <resource>
        <targetPath>META-INF/plexus</targetPath>
        <filtering>false</filtering>
        <directory>${project.basedir}/src/main/plexus</directory>
        <includes>
          <include>configuration.xml</include>
        </includes>
        <excludes>
          <exclude>**/*.properties</exclude>
        </excludes>
      </resource>
    </resources>
    <testResources>
      ...
    </testResources>
    ...
  </build>
</project>
```

- **resources**: 是一个资源元素列表，每个元素描述了包含在此项目中的文件的内容和位置。
- **targetPath**: 指定将构建的一组资源放置在其中的目录结构。目标路径默认为基本目录。将打包在 JAR 中的资源指定的常见目标路径是 META-INF。
- **filtering**: 是 `true` 或 `false` ，表示是否为此资源启用过滤。请注意，不必为过滤文件 `*.properties` 定义过滤才能发生-资源也可以使用默认在 POM 中定义的属性（例如${project.version}），通过使用“-D”标志传递到命令行（例如，“ `-Dname` = `value` ”）或由属性元素明确定义。
- **directory**: 此元素的值定义了资源的位置。构建的默认目录是 `${project.basedir}/src/main/resources` 。
- **includes**: 一组文件模式，指定要包括在指定目录下的资源文件，使用*作为通配符。
- **excludes**:与 `includes` 相同的结构，但指定要忽略哪些文件。在 `include` 和 `exclude` 之间发生冲突时， `exclude` 获胜。
- **testResources**: `testResources` 元素块包含 `testResource` 个元素。它们的定义类似于 `resource` 元素，但在测试阶段自然使用。唯一的区别是项目的默认（Super POM 定义的）测试资源目录是 `${project.basedir}/src/test/resources` 。测试资源不会被部署。

#### Plugins 插件

``` xml
<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">
  <build>
    ...
    <plugins>
      <plugin>
        <groupId>org.apache.maven.plugins</groupId>
        <artifactId>maven-jar-plugin</artifactId>
        <version>2.6</version>
        <extensions>false</extensions>
        <inherited>true</inherited>
        <configuration>
          <classifier>test</classifier>
        </configuration>
        <dependencies>...</dependencies>
        <executions>...</executions>
      </plugin>
    </plugins>
  </build>
</project>
```

除了`groupId:artifactId:version` 的范围，还有配置插件或构建与之交互的元素。

- **extensions**: `true` 或 `false` ，是否加载此插件的扩展。默认情况下为 false。
- **inherited**: `true` 或 `false` ，无论是否应将此插件配置应用于继承自此插件的 POM。默认值为 `true` 。
- **configuration**: 这是针对单个插件的特定配置。不深入探讨插件工作机制，可以说插件 Mojo 可能期望的任何属性（这些是 Java Mojo bean 中的 getter 和 setter）可以在这里指定。在上面的示例中，我们将分类器属性设置为测试在 `maven-jar-plugin` 的 Mojo 中。值得注意的是，无论配置元素位于 POM 的任何位置，都旨在将值传递给另一个底层系统，例如插件。换句话说： `configuration` 元素中的值从未被 POM 模式明确要求，但插件目标有权要求配置值。
- **dependencies**: 依赖项在 POM 中经常出现，并且是所有插件元素块下的一个元素。这些依赖项具有与基本构建下相同的结构和功能。在这种情况下的主要区别是，它们不再作为项目的依赖项应用，而是作为它们所在插件的依赖项应用。这样做的好处是可以通过 `exclusions` 删除一个未使用的运行时依赖项来修改插件的依赖项列表，或者修改所需依赖项的版本。有关更多信息，请参见上面的依赖项。
- **executions**: 重要的是要记住一个插件可能有多个目标。每个目标可能有单独的配置，甚至可能将插件的目标绑定到完全不同的阶段。 `executions` 配置插件目标的 `execution` 。
	- **id**:它指定了这个执行块在所有其他执行块之间的位置。当运行该阶段时，它将显示为： `[plugin:goal execution: id]` 。在这个示例中： `[antrun:run execution: echodir]`
	- **goals**: 与所有复数化的 POM 元素一样，这包含了一系列单一元素。在这种情况下，这是一个由此 `execution` 块指定的插件 `goals` 列表。
	- **phase**: 这是目标列表将执行的阶段。这是一个非常强大的选项，允许将任何目标绑定到构建生命周期中的任何阶段，从而改变 Maven 的默认行为。
	- **inherited**: 与上面的 `inherited` 元素一样，将其设置为 false 将阻止 Maven 将此执行传递给其子级。此元素仅对父 POM 有意义。
	- **configuration**:与上述相同，但将配置限制在此特定目标列表中，而不是插件下的所有目标。
``` xml
<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">
  ...
  <build>
    <plugins>
      <plugin>
        <artifactId>maven-antrun-plugin</artifactId>
        <version>1.1</version>
        <executions>
          <execution>
            <id>echodir</id>
            <goals>
              <goal>run</goal>
            </goals>
            <phase>verify</phase>
            <inherited>false</inherited>
            <configuration>
              <tasks>
                <echo>Build Dir: ${project.build.directory}</echo>
              </tasks>
            </configuration>
          </execution>
        </executions>
      </plugin>
    </plugins>
  </build>
</project>
```

### Reporting
报告包含与 `site` 生成阶段特定对应的元素。某些 Maven 插件可以生成在报告元素下定义和配置的报告，例如：生成 Javadoc 报告。

``` xml
<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">
  ...
  <reporting>
    <outputDirectory>/home/jenkins/82467a7c/workspace/aven_maven-box_maven-site_master/target/site</outputDirectory>
    <plugins>
      <plugin>
        <artifactId>maven-project-info-reports-plugin</artifactId>
        <version>2.0.1</version>
        <reportSets>
          <reportSet></reportSet>
        </reportSets>
      </plugin>
    </plugins>
  </reporting>
  ...
</project>
```
在报告的情况下，默认情况下输出目录为 `${project.basedir}/target/site` 。


## Repositories

Repository是 Maven 社区中最强大的功能之一。默认情况下，Maven 在 https://repo.maven.apache.org/maven2/中搜索中央存储库。可以在 pom.xml 的`repositories`元素中配置其Repository。

``` xml
<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">
  ...
  <repositories>
    <repository>
      <releases>
        <enabled>false</enabled>
      </releases>
      <snapshots>
        <enabled>true</enabled>
        <updatePolicy>always</updatePolicy>
        <checksumPolicy>fail</checksumPolicy>
      </snapshots>
      <name>Nexus Snapshots</name>
      <id>snapshots-repo</id>
      <url>https://oss.sonatype.org/content/repositories/snapshots</url>
      <layout>default</layout>
    </repository>
  </repositories>
  <pluginRepositories>
    ...
  </pluginRepositories>
  ...
</project>
```
- **releases**, **snapshots**: 这些是每种构件类型（发布或快照）的策略。有了这两组策略，POM 可以在单个存储库中独立于另一种类型改变每种类型的策略。例如，一个人可能决定仅允许快照下载，可能是为了开发目的
- **enabled**:`true` 或 `false` 用于确定此存储库是否已启用相应类型（ `releases` 或 `snapshots` ）。默认情况下为 `true`
- **updatePolicy**:updatePolicy: 此元素指定 Maven 尝试从远程存储库更新本地存储库的频率。
	-  `always` 
	- `daily` （默认值）
	- `interval:X` （其中 X 是以分钟为单位的整数）
	-  `never` （仅在本地存储库中尚不存在时下载）。由于这会影响到构件和元数据（预计在 Maven 4 中进行更改），请注意 `never` ，因为元数据会随时间变化（即使是对于发布存储库）。
- **checksumPolicy**:当 Maven 将文件部署到存储库时，它还会部署相应的校验和文件。您可以选择在缺少或不正确的校验和时 `ignore` ， `fail` 或 `warn` 。默认值为 `warn` 。
- **id**:存储库 ID 是必需的，将存储库与来自 `settings.xml` 的服务器连接起来。其默认值为 `default` 。该 ID 还用于本地存储库元数据中存储来源。
- **name**: 存储库的可选名称。在发出与此存储库相关的日志消息时用作标签。
- **layout**: 在上述存储库描述中，提到它们都遵循一个共同的布局。这基本上是正确的。Maven 2 引入的布局是 Maven 2 和 3 都使用的存储库的默认布局。然而，Maven 1.x 有一个不同的布局。使用此元素指定它是 `default` 还是 `legacy` 。其默认值为 `default` 。

## Profiles 配置
POM 4.0 的一个新功能是项目能够根据构建环境来更改设置。 `profile` 元素包含可选的激活（配置文件触发器）和如果激活了该配置文件，则要对 POM 进行的更改集。例如，为测试环境构建的项目可能指向与最终部署不同的数据库。或者根据使用的 JDK 版本，依赖项可能从不同的存储库中提取。配置文件的元素如下：
``` xml
<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">
  ...
  <profiles>
    <profile>
      <id>test</id>
      <activation>...</activation>
      <build>...</build>
      <modules>...</modules>
      <repositories>...</repositories>
      <pluginRepositories>...</pluginRepositories>
      <dependencies>...</dependencies>
      <reporting>...</reporting>
      <dependencyManagement>...</dependencyManagement>
      <distributionManagement>...</distributionManagement>
    </profile>
  </profiles>
</project>
```
### Activation 激活

Activatio是配置文件的关键。配置文件的能力来自于它在特定情况下仅能修改基本 POM 的能力。这些情况是通过 `activation` 元素指定的。

``` xml
<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">
  ...
  <profiles>
    <profile>
      <id>test</id>
      <activation>
        <activeByDefault>false</activeByDefault>
        <jdk>1.5</jdk>
        <os>
          <name>Windows XP</name>
          <family>Windows</family>
          <arch>x86</arch>
          <version>5.1.2600</version>
        </os>
        <property>
          <name>sparrow-type</name>
          <value>African</value>
        </property>
        <file>
          <exists>${basedir}/file2.properties</exists>
          <missing>${basedir}/file1.properties</missing>
        </file>
      </activation>
      ...
    </profile>
  </profiles>
</project>
```
**在 Maven 3.2.2 之前，激活发生在满足了一个或多个指定条件时。当遇到第一个积极结果时，处理停止并将配置文件标记为活动状态。自 Maven 3.2.2 起，激活发生在满足了所有指定条件**

- **activeByDefault**:默认情况下为 `false` 。布尔标志，用于确定配置文件是否默认处于活动状态。仅当没有其他配置文件通过命令行显式激活 `settings.xml` 或通过其他激活器隐式激活时，才会评估此标志，否则它将不起作用。
- **jdk**:`activation` 在 `jdk` 元素中内置了一个以 Java 为中心的检查。该值是以下三种类型之一：
	- 根据 maven-enforcer-plugin 的定义，如果值以 `[` 或 `(` 开头，则版本范围
	- 如果值以 `!` 开头，则为否定前缀
	- 所有其他情况的（非否定）前缀
- **os**:`os` 元素可能需要一些具有特定值的特定操作系统属性。每个值可能以 `!` 开头
	- **name**: 与系统属性 `os.name` 匹配
	- **family**: 与从其他 os.*系统属性派生的家庭进行匹配
	- **arch**: 与系统属性 `os.arch` 匹配
	- **version**: 与系统属性 `os.version` 匹配。
- **property**: 如果 Maven 检测到相应的 `name=value` 对的系统属性或 CLI 用户属性（可以在 POM 中通过 `${name}` 进行解引用的值），并且与给定值（如果给定）匹配，则 `profile` 将被激活。
- **file**:最后，给定的文件名可能通过文件的 `profile` 激活 `existence` ，或者如果它是 `missing` 。注意：此元素的插值仅限于 `${basedir}` ，系统属性和请求属性。

`activation` 元素不是激活 `profile` 的唯一方式。 `settings.xml` 文件的 `activeProfile` 元素可能包含配置文件的 `id`