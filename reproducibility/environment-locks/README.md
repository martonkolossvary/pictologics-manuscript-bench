# Native adapter environment locks

These files bind the complete `pip freeze --all` package sets, including the
installer, for Windows AMD64 and macOS arm64. Their byte and canonical freeze
SHA-256 hashes are recorded in `configs/adapters/*.yaml`. `bench env create`
installs the matching lock; `bench env verify` rejects package drift and checks
the environment's saved provenance. An environment must be recreated on the
destination machine; copying a virtual environment between operating systems
does not recreate its native dependencies.

## Pictologics 0.7.1 update, 2026-10-12

The current release was checked using the live [PyPI JSON endpoint](https://pypi.org/pypi/pictologics/json)
and [GitHub release](https://github.com/martonkolossvary/pictologics/releases/tag/v0.7.1).
The PyPI wheel was uploaded on 2026-10-08 and the GitHub tag resolves to
`f2fdb963b17b483f339ff547c23fa519bf4850b9`.

The installer enforces wheel SHA-256
`aa9bf3cff2ea22012732cb40f2ada17d49d0a87643c80db66667a10284537bfc`.
The lock records the direct wheel URL in the form produced by `pip freeze`;
the profile adds the hash when installing it.

The update preserves existing transitive package versions and adds
`python-gdcm==3.2.6`, a new required dependency. Existing optional/legacy
packages remain installed to avoid unrelated environment changes. The
Windows lock has 29 entries and was installed and verified natively with
Python 3.12.10, `pip check`, import probes, and all six untimed native workload
smokes on a small synthetic image. This checks installation and API
compatibility; it is not an IBSI compliance result or a performance result.

The macOS arm64 lock has 27 entries. Its complete package set resolves with
`uv==0.8.24`, `--python-version 3.12`, and
`--python-platform aarch64-apple-darwin`. The new GDCM dependency publishes
CPython 3.12 wheels for both native target platforms. This upgrade has not yet
been executed on a local Mac. The `native-adapters` CI jobs install all five
native environments and exercise every supported benchmark workload on
Windows and macOS arm64; their passing results must be checked before claiming
the upgrade has passed on both systems. Each publication host must additionally
run its own frozen input audit and host qualification.

The macOS CI runner uses `macos-15`, documented as arm64 in the
[GitHub runner reference](https://docs.github.com/en/actions/reference/runners/github-hosted-runners).
The workflow checks that every adapter has a matching native lock, so a future
runner architecture change cannot silently switch to unlocked dependencies.
Intel macOS and Windows ARM64 have no checked-in qualification locks here.

From the repository root, after environment creation:

```text
poetry run python -m bench.cli env verify
poetry run python -m scripts.smoke_adapter_environments
```

Use `bench env create --profiles pictologics --force` to replace a pre-upgrade
Pictologics environment. Finish or preserve any runs bound to the previous
environment before rebuilding it; benchmark provenance prevents combining
different Pictologics releases in a single resumed run.
