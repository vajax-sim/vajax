# Release VAJAX

Prepare and execute a release for VAJAX. The argument should be a version number (e.g. `0.2.0`) or `patch`/`minor`/`major` to auto-bump.

## Steps

1. **Determine version**: Parse the argument to get the target version.
   - If `patch`/`minor`/`major`: read the latest git tag (`git tag -l 'v*' --sort=-version:refname | head -1`), bump accordingly
   - If a version number: use it directly
   - Confirm the version with the user before proceeding

2. **Pre-flight checks**:
   - Verify we're on `main` branch with a clean working tree (`git status`)
   - Verify all CI checks pass on HEAD (`gh run list --branch main --limit 3`)
   - Run local tests: `JAX_PLATFORMS=cpu uv run pytest tests/test_vacask_suite.py tests/test_vacask_jax.py -v --tb=short`
   - Run pyright: `uv run pyright vajax/`
   - Run ruff: `uv run ruff check .`

3. **Generate changelog**:
   - List commits since last tag: `git log --oneline $(git describe --tags --abbrev=0)..HEAD`
   - Categorise into: Features, Fixes, CI/Infrastructure, Documentation, Other
   - Present the changelog to the user for review/editing

4. **Create the release**:
   - Create and push the tag: `git tag v{version} && git push origin v{version}`
   - This triggers the release workflow which handles:
     - Version patching across sub-packages (via `scripts/set_release_version.py`)
     - Building wheels (vajax, openvaf-py, osdi-py, umfpack-jax) on Linux/macOS/Windows
     - Publishing to TestPyPI then PyPI
     - Creating GitHub Release with auto-generated notes
     - Deploying docs to GitHub Pages
     - Updating Homebrew tap

5. **Monitor the release**:
   - Watch the release workflow: `gh run watch` on the triggered Release workflow
   - Report status of each job (check-ci, build, test-install, publish, github-release, docs, homebrew)

## Packages released

| Package | Type | Version source |
|---------|------|----------------|
| vajax | Pure Python | hatch-vcs (from git tag) |
| openvaf-py | Rust/maturin | Patched by set_release_version.py |
| osdi-py | Rust/maturin | Patched by set_release_version.py |
| umfpack-jax | C++/scikit-build | Patched by set_release_version.py |

## Important notes

- vajax version is automatic via `hatch-vcs` -- no files need manual version bumps
- Sub-package versions are patched by CI via `scripts/set_release_version.py`
- The release workflow waits for Lint and Tests workflows to pass before publishing
- TestPyPI publish happens on every push to main (as RC versions from git describe)
- PyPI publish only happens on tag pushes
