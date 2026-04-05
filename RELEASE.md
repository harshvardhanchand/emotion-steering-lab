# Release Process

This project uses Semantic Versioning.

## Versioning Policy

- `MAJOR`: incompatible schema or CLI contract changes.
- `MINOR`: backward-compatible features.
- `PATCH`: backward-compatible fixes.

## How to Cut a Release

1. Ensure CI is green on `main`.
2. Update `CHANGELOG.md` under a new version heading.
3. Create and push a tag:

```bash
git tag v0.1.1
git push origin v0.1.1
```

4. The `Release` workflow will:
- run contract checks,
- run tests,
- build package artifacts,
- publish GitHub Release notes and assets.
