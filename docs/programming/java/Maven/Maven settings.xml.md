`settings.xml` 文件中的 `settings` 元素包含用于定义配置 Maven 执行方式的值的元素,这些值包括本地仓库位置、备用远程仓库服务器和认证信息。

两个位置可以放settings.xml
- Maven 安装路径： `${maven.home}/conf/settings.xml`
- 用户的安装路径: `${user.home}/.m2/settings.xml`

``` xml
<settings xmlns="http://maven.apache.org/SETTINGS/1.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:schemaLocation="http://maven.apache.org/SETTINGS/1.0.0 https://maven.apache.org/xsd/settings-1.0.0.xsd">
  <localRepository/>
  <interactiveMode/>
  <offline/>
  <pluginGroups/>
  <servers/>
  <mirrors/>
  <proxies/>
  <profiles/>
  <activeProfiles/>
</settings>
```

`settings.xml` 的内容可以使用以下表达式进行插值：

1. `${user.home}`  和所有其他系统属性
2. `${env.HOME}` 等用于环境变量

## Setting Simple Values
- **localRepository**: 此值是此构建系统的本地存储库的路径。默认值为 `${user.home}/.m2/repository` 。此元素对于主构建服务器特别有用，允许所有已登录用户从共同的本地存储库构建。
- **interactiveMode**:  `true` 如果 Maven 应尝试与用户交互以获取输入, `false` 如果不应。默认为 `true` 。
- **offline**: `true` 如果此构建系统应在离线模式下运行，默认为 `false` 。此元素对于无法连接到远程存储库的构建服务器非常有用，可能是因为网络设置或安全原因
## Plugin Groups

```xml
<settings xmlns="http://maven.apache.org/SETTINGS/1.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:schemaLocation="http://maven.apache.org/SETTINGS/1.0.0 https://maven.apache.org/xsd/settings-1.0.0.xsd">
  ...
  <pluginGroups>
    <pluginGroup>org.eclipse.jetty</pluginGroup>
  </pluginGroups>
  ...
</settings>
```

## Serves
下载和部署的存储库由 POM 的 `repositories` 和 `distributionManagement` 元素定义。然而，诸如 `username` 和 `password` 等特定设置不应与 `pom.xml` 一起分发。这种类型的信息应存在于构建服务器中的 `settings.xml` 。

```xml
<settings xmlns="http://maven.apache.org/SETTINGS/1.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:schemaLocation="http://maven.apache.org/SETTINGS/1.0.0 https://maven.apache.org/xsd/settings-1.0.0.xsd">
  ...
  <servers>
    <server>
      <id>server001</id>
      <username>my_login</username>
      <password>my_password</password>
      <privateKey>${user.home}/.ssh/id_dsa</privateKey>
      <passphrase>some_passphrase</passphrase>
      <filePermissions>664</filePermissions>
      <directoryPermissions>775</directoryPermissions>
      <configuration></configuration>
    </server>
  </servers>
  ...
</settings>
```

- **id**: 这是服务器的 ID（而不是要登录的用户的 ID），与 Maven 尝试连接的存储库/镜像的 `id` 元素匹配。
- **username**, **password**: 这些元素作为一对出现，表示登录到此服务器所需的登录名和密码。
- **privateKey**, **passphrase**:与前两个元素一样，此对指定了私钥的路径（默认为 `${user.home}/.ssh/id_dsa` ）和 `passphrase` ，如果需要的话。 `passphrase` 和 `password` 元素可能会在将来外部化，但目前必须在 `settings.xml` 文件中设置为明文。
- **filePermissions**, **directoryPermissions**:文件权限，目录权限：当部署时创建存储库文件或目录时，这些是要使用的权限。每个的合法值是对应于*nix 文件权限的三位数字，例如 664 或 775。

## Mirrors
镜像是中央仓库或远程仓库的复制品，通常用于提高下载速度、减少网络延迟或绕过某些网络限制。配置镜像可以让Maven将所有仓库请求重定向到配置的镜像地址，从而加快依赖下载的速度和稳定性。
```xml
<settings xmlns="http://maven.apache.org/SETTINGS/1.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:schemaLocation="http://maven.apache.org/SETTINGS/1.0.0 https://maven.apache.org/xsd/settings-1.0.0.xsd">
  ...
  <mirrors>
    <mirror>
      <id>planetmirror.com</id>
      <name>PlanetMirror Australia</name>
      <url>http://downloads.planetmirror.com/pub/maven2</url>
      <mirrorOf>central</mirrorOf>
    </mirror>
  </mirrors>
  ...
</settings>
```

- **id**, **name**: 此镜像的唯一标识符和用户友好名称。 `id` 用于区分 `mirror` 元素，并在连接到镜像时从 `<servers>` 部分选择相应的凭据。
- **url**: 此镜像的基本 URL。构建系统将使用此 URL 连接到存储库，而不是原始存储库 URL。
- **mirrorOf**: 这是镜像的存储库的 `id` 。例如，要指向 Maven `central` 存储库的镜像（ `https://repo.maven.apache.org/maven2/` ），请将此元素设置为 `central` 。还可以进行更高级的映射，如 `repo1,repo2` 或 `*,!inhouse` 。这不能匹配镜像 `id` 。
## Proxies
```xml
<settings xmlns="http://maven.apache.org/SETTINGS/1.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:schemaLocation="http://maven.apache.org/SETTINGS/1.0.0 https://maven.apache.org/xsd/settings-1.0.0.xsd">
  ...
  <proxies>
    <proxy>
      <id>myproxy</id>
      <active>true</active>
      <protocol>http</protocol>
      <host>proxy.somewhere.com</host>
      <port>8080</port>
      <username>proxyuser</username>
      <password>somepassword</password>
      <nonProxyHosts>*.google.com|ibiblio.org</nonProxyHosts>
    </proxy>
  </proxies>
  ...
</settings>
```

- **id**: 此代理的唯一标识符。这用于区分 `proxy` 元素。
- **active**: `true` 如果此代理处于活动状态。这对于声明一组代理很有用，但一次只能有一个代理处于活动状态。
- **protocol**, **host**, **port**: 协议，主机，端口：代理的 `protocol://host:port` ，分隔为离散元素。
- **username**, **password**: 这些元素作为一对出现，表示登录到此代理服务器所需的登录名和密码。
- **nonProxyHosts**: 这是一个不应该被代理的主机列表。列表的分隔符是代理服务器的预期类型；上面的示例是以管道分隔的，逗号分隔也很常见。
## Profiles
`settings.xml` 中的 `profile` 元素是 `pom.xml` `profile` 元素的截断版本。它由 `activation` 、 `repositories` 、 `pluginRepositories` 和 `properties` 元素组成。 `profile` 元素仅包括这四个元素，因为它们关注整个构建系统（这是 `settings.xml` 文件的作用），而不关注个别项目对象模型设置。

如果一个配置文件从 `settings` 处激活，则其值将覆盖 POM 或 `profiles.xml` 文件中具有相同 ID 的配置文件的任何值。

### Activation

```xml
<settings xmlns="http://maven.apache.org/SETTINGS/1.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:schemaLocation="http://maven.apache.org/SETTINGS/1.0.0 https://maven.apache.org/xsd/settings-1.0.0.xsd">
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
          <name>mavenVersion</name>
          <value>2.0.3</value>
        </property>
        <file>
          <exists>${basedir}/file2.properties</exists>
          <missing>${basedir}/file1.properties</missing>
        </file>
      </activation>
      ...
    </profile>
  </profiles>
  ...
</settings>
```

### Properties
Maven 属性是值占位符，类似于 Ant 中的属性。通过使用符号 `${X}` ，可以在 POM 中的任何地方访问它们的值，其中 `X` 是属性。它们有五种不同的样式，都可以从 `settings.xml` 文件中访问：

1. `env.X`: 在变量前加上“env.”将返回 shell 的环境变量。例如， `${env.PATH}` 包含$path 环境变量（在 Windows 中为 `%PATH%` ）。
2. `project.x`: POM 中的点（.）标记路径将包含相应元素的值。例如： `<project><version>1.0</version></project>` 可通过 `${project.version}` 访问。
3. `settings.x`: 在 `settings.xml` 中以点（.）标记的路径将包含相应元素的值。例如： `<settings><offline>false</offline></settings>` 可通过 `${settings.offline}` 访问。
4. Java System Properties: 通过 `java.lang.System.getProperties()` 访问的所有属性都可以作为 POM 属性使用，例如 `${java.home}` 。
5. `x`: 可以在元素或外部文件中设置，该值可以用作 `${someVar}` 。

```xml
<settings xmlns="http://maven.apache.org/SETTINGS/1.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:schemaLocation="http://maven.apache.org/SETTINGS/1.0.0 https://maven.apache.org/xsd/settings-1.0.0.xsd">
  ...
  <profiles>
    <profile>
      ...
      <properties>
        <user.install>${user.home}/our-project</user.install>
      </properties>
      ...
    </profile>
  </profiles>
  ...
</settings
```

### Repositories
Repository是远程项目集合，Maven 用它来填充构建系统的本地存储库。Maven 从这个本地存储库调用插件和依赖项。不同的远程存储库可能包含不同的项目，在活动配置文件下，它们可能被搜索以匹配发布或快照工作

```xml
<settings xmlns="http://maven.apache.org/SETTINGS/1.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:schemaLocation="http://maven.apache.org/SETTINGS/1.0.0 https://maven.apache.org/xsd/settings-1.0.0.xsd">
  ...
  <profiles>
    <profile>
      ...
      <repositories>
        <repository>
          <id>codehausSnapshots</id>
          <name>Codehaus Snapshots</name>
          <releases>
            <enabled>false</enabled>
            <updatePolicy>always</updatePolicy>
            <checksumPolicy>warn</checksumPolicy>
          </releases>
          <snapshots>
            <enabled>true</enabled>
            <updatePolicy>never</updatePolicy>
            <checksumPolicy>fail</checksumPolicy>
          </snapshots>
          <url>http://snapshots.maven.codehaus.org/maven2</url>
          <layout>default</layout>
        </repository>
      </repositories>
      <pluginRepositories>
        <pluginRepository>
          <id>myPluginRepo</id>
          <name>My Plugins repo</name>
          <releases>
            <enabled>true</enabled>
          </releases>
          <snapshots>
            <enabled>false</enabled>
          </snapshots>
          <url>https://maven-central-eu....com/maven2/</url>
        </pluginRepository>
      </pluginRepositories>
      ...
    </profile>
  </profiles>
  ...
</settings>
```

- **releases**, **snapshots**: 这些是每种构件类型（发布或快照）的策略。有了这两组策略，POM 可以在单个存储库中独立于另一种类型改变每种类型的策略。例如，一个人可能决定仅允许快照下载，可能是为了开发目的。
- **enabled**:  `true` 或 `false` 表示该存储库是否已启用相应类型（ `releases` 或 `snapshots` ）。
- **updatePolicy**:  此元素指定更新应尝试发生的频率。 Maven 将比较本地 POM 的时间戳（存储在存储库的 maven-metadata 文件中）与远程时间戳。 选择项为： `always` ， `daily` （默认）， `interval:X` （其中 X 是以分钟为单位的整数）或 `never` 。
- **checksumPolicy**:  当 Maven 将文件部署到存储库时，它还会部署相应的校验和文件。您可以选择在缺少或不正确的校验和时 `ignore` ， `fail` 或 `warn` 。
- **layout**: 在上述存储库描述中提到，它们都遵循一个共同的布局。这基本上是正确的。Maven 2 有一个默认的存储库布局；然而，Maven 1.x 有一个不同的布局。使用此元素来指定它是 `default` 还是 `legacy` 。

## Active Profiles

```xml
<settings xmlns="http://maven.apache.org/SETTINGS/1.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:schemaLocation="http://maven.apache.org/SETTINGS/1.0.0 https://maven.apache.org/xsd/settings-1.0.0.xsd">
  ...
  <activeProfiles>
    <activeProfile>env-test</activeProfile>
  </activeProfiles>
</settings>
```
其中包含一组 `activeProfile` 元素，每个元素的值为 `profile` `id` 。任何定义为 `activeProfile` 的 `profile` `id` 都将处于活动状态，而不受任何环境设置的影响。如果找不到匹配的配置文件，则不会发生任何事情。例如，如果 `env-test` 是 `activeProfile` ，则将激活一个在 `pom.xml` （或 `profile.xml` 与相应 `id` ）中的配置文件。如果找不到这样的配置文件，则执行将继续正常进行。