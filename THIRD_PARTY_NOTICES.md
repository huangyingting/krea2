# Third-party notices

The standalone runtime does not import or launch an external workflow engine.
Some algorithms and compatibility behavior were derived from these upstream
projects and retain their original licensing requirements:

- ByteDance SeedVR / SeedVR2 — Apache-2.0. The retained license is at
  `src/krea2pipe/seedvr2/LICENSE`.
- Krea 2 reference implementation — <https://github.com/krea-ai/krea-2>.
- ComfyUI and related workflow extensions used during the original numerical
  validation — GPL-3.0 and their respective repository licenses.

Development-only reference scripts under `tools/` may import a separately
installed reference runtime. They are never imported by `krea2pipe`.
