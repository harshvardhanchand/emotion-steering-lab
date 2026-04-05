# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project follows Semantic Versioning.

## [Unreleased]

### Changed

- Removed legacy v1 red-team evaluation pipeline.
- Repositioned project as a steering-only framework.

### Added

- Steering-only CLI with command groups:
  - `art data generate`
  - `art probe train`
  - `art steer run`
  - `art steer alpha-grid`
- Synthetic probe data generation pipeline.
- Probe training pipeline producing `probe_artifact.json`.
- Steering run pipeline producing `steering_results.jsonl`.
- Steering profile generation (`steering_profile.json`).
- New schema contract: `steering_profile.schema.json`.
- Steering-focused tests and contract checks.
