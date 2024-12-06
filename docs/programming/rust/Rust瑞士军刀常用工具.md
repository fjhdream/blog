## cargo expand
[dtolnay/cargo-expand: Subcommand to show result of macro expansion (github.com)](https://github.com/dtolnay/cargo-expand)
rust macro is `code generation`

## cargo edit
[killercup/cargo-edit: A utility for managing cargo dependencies from the command line. (github.com)](https://github.com/killercup/cargo-edit)


This tool extends [Cargo](http://doc.crates.io/) to allow you to add, remove, and upgrade dependencies by modifying your `Cargo.toml` file from the command line.

### useage

1. cargo add
2. cargo rm
3. cargo upgrade
``` shell
$ cargo-upgrade upgrade --help
Upgrade dependency version requirements in Cargo.toml manifest files

Usage: cargo upgrade [OPTIONS]

Options:
      --dry-run               Print changes to be made without making them
      --manifest-path <PATH>  Path to the manifest to upgrade
      --rust-version <VER>    Override `rust-version`
      --ignore-rust-version   Ignore `rust-version` specification in packages
      --offline               Run without accessing the network
      --locked                Require `Cargo.toml` to be up to date
  -v, --verbose...            Use verbose output
  -Z <FLAG>                   Unstable (nightly-only) flags
  -h, --help                  Print help
  -V, --version               Print version

Version:
      --compatible [<allow|ignore>]    Upgrade to latest compatible version [default: allow]
  -i, --incompatible [<allow|ignore>]  Upgrade to latest incompatible version [default: ignore]
      --pinned [<allow|ignore>]        Upgrade pinned to latest incompatible version [default:
                                       ignore]

Dependencies:
  -p, --package <PKGID[@<VERSION>]>  Crate to be upgraded
      --exclude <PKGID>              Crates to exclude and not upgrade
      --recursive [<true|false>]     Recursively update locked dependencies

```

4. cargo set-version