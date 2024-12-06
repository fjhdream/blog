---
title: 提高 rust 编译速度技巧
categories:
  - 技术学习
tags:
  - Rust
halo:
  site: http://205.234.201.223:8090
  name: dcaf1236-3ec8-487b-b8ce-56752a10d966
  publish: false
---
## 使用cargo check而不是cargo build
``` bash
# Slow 🐢
cargo build

# Fast 🐇 (2x-3x speedup)
cargo check
```


![扫描指令对比](http://picbed.fjhdream.cn/202401311702858.png)

**技巧**: 使用 `cargo watch -c`  , 这样，每当你更改一个文件时，它就会 `cargo check` 。 也会清理屏幕

## 删除未使用的依赖项

``` bash
cargo install cargo-machete && cargo machete
```

## 更新依赖

1.  运行 `cargo update` 以更新到最新的兼容版本。
2. 运行 `cargo outdated -wR` 以查找更新的、可能不兼容的依赖项。根据需要更新这些依赖项并修复代码。
3. 运行 `cargo tree --duplicate` 以查找存在多个版本的依赖项。
4. 使用 `cargo audit` 来获取有关需要解决的漏洞或需要替换的废弃包的通知

## 查找编译最慢的 crate

```sh
cargo build --timings
```

![示例](http://picbed.fjhdream.cn/202401311714438.png)

这个图表中的**红线**显示了当前等待编译的单位（货箱）的数量（被另一个货箱阻塞）。如果有大量的货箱在一个货箱上出现瓶颈，应该集中精力改进那个货箱，以提高并行性。

## 配置查看具体编译时间
可以使用 `cargo rustc -- -Zself-profile` 对其进行分析。生成的跟踪文件可以使用火焰图或Chromium分析器进行可视化。

另一个重要的指标是 `cargo-llvm-lines` ，它显示了生成的代码行数以及最终二进制文件中每个通用函数的副本数量。这可以帮助您确定哪些函数在编译过程中消耗最多。

## 替换重的crate

`cargo-bloat` 还有一个 `--time` 标志，显示每个箱子的构建时间。非常方便！

|Crate  |Alternative  |
|---|---|
|[serde](https://github.com/bnjbvr/cargo-machete)|[miniserde](https://github.com/dtolnay/miniserde), [nanoserde](https://github.com/not-fl3/nanoserde)  |
|[reqwest](https://github.com/seanmonstar/reqwest) |[ureq](https://github.com/algesten/ureq)|
|[clap](https://github.com/clap-rs/clap) |[lexopt](https://github.com/blyxxyz/lexopt)|

## 禁用未使用的依赖

```sh
cargo install cargo-features-manager
cargo features prune
```

## 使用sccache进行缓存依赖

 [sccache](https://github.com/mozilla/sccache)  它将编译的crate缓存起来，以避免重复编译。

## Cranelift：Rust的替代编译器

##  切换到更快的链接器

检查瓶颈
```
cargo clean
cargo +nightly rustc --bin <your_binary_name> -- -Z time-passes
```

|Linker|Platform|Production Ready|Description|
|---|---|---|---|
|[`lld`](https://lld.llvm.org/)|Linux/macOS|Yes|Drop-in replacement for system linkers|
|[`mold`](https://github.com/rui314/mold)|Linux|[Yes](https://github.com/bluewhalesystems/sold)|Optimized for Linux|
|[`zld`](https://github.com/michaeleisel/zld)|macOS|No (deprecated)|Drop-in replacement for Apple's `ld` linker|

## macOS: 更快的增量调试构建

添加到 Cargo.toml
```toml
[profile.dev]
split-debuginfo = "unpacked"
```