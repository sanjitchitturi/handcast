# Contributing

Follow the README setup instructions and keep each pull request focused on one numerical,
data-contract, or interface change.

## Verification

Run from the repository root:

```sh
python -m pytest tests
```

Add deterministic regression coverage for geometry, dynamics, and tolerance changes.
State coordinate-frame assumptions, units, robot configuration, and fixture provenance.
Do not describe simulation checks as validation for hardware operation.

Keep private video, biometric landmarks, robot credentials, and generated artifacts out of
commits. Follow [SECURITY.md](SECURITY.md) for security-sensitive reports.
