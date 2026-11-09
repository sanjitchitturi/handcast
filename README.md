# HandCast

Analyze hand trajectories and retarget video landmarks for robot simulation. The numerical modules cover pose interpolation, phase segmentation and velocity, acceleration and jerk checks.

## Quick start

Run the numerical checks from the repository root:

```sh
python -m pip install numpy scipy osqp pytest
python -m pytest tests
```

For the video pipeline, use Python 3.11 and a MediaPipe build that includes Hands. Install the project dependencies, including OSQP, then run `handcast run --help` for video, robot and output options. Video ingestion processes up to 120 sampled frames per run. Checkpoints and robot descriptions should come from trusted sources. Inspect the output in simulation before considering hardware use. The feasibility checker reports the largest absolute component of each translational derivative and applies the same phase-specific limits as the per-axis QP. It does not establish Euclidean speed or hardware safety. Projection raises if OSQP is missing or the solver fails.

## Synthetic trajectory check

A synthetic straight trajectory with 30 poses, identity rotations and 5 cm of travel every 50 ms produced an observed speed of 1.00 m/s. With the default 0.60 m/s limit, the checker flagged every velocity interval. Acceleration and jerk stayed within limits.

## Verification

Run `python -m pytest tests` to check pose transforms, phase segmentation, and trajectory limits.

## License

[MIT License](LICENSE) for the project code. Third-party software, models, and datasets retain their own terms.
