# Quadruple pendulum web

A static, browser-only version of the quadruple inverted pendulum simulation.
It is a Vite + TypeScript app with no backend: physics and control run in the
browser, and the output is plain static files.

All commands below run inside `web/`.

## Local development

```bash
npm install
npm run dev        # local dev server with hot reload
npm run preview    # serve the production build from web/dist/
```

## Tests

```bash
npm test           # vitest parity tests against Python golden traces (test/golden/)
```

## Build

```bash
npm run build      # type-checks, then writes static files to web/dist/
```

`vite.config.ts` sets `base: "./"`, so `dist/` works from any URL sub-path.

## Presets

Presets live in `public/presets/*.json`. They are generated from the Python
package. From the repo root, with the project venv active:

```bash
python scripts/export_web.py
```

This also refreshes `web/test/golden/*`. Rerun it after changing
`src/quad_pendulum/params.py` or the controller, then run `npm test` and
`npm run build` again.

Each preset exports `configurations: { UUUU: { controller, unstablePoles }, ... }`
for all `2^n` up/down combinations. Every controller includes `configuration`
and `thetaEq` (absolute target angles in radians). The top-level `controller`
and `unstablePoles` remain the all-up default. Letters run **bottom link first**:
`UUUD` has its top link hanging down; `DDUU` has its bottom two down.

The simulation starts with every link hanging down at rest; Reset returns to
that state and centres the target. The configuration picker is a **go to**
control: choosing a target centres and settles the cart, tracks a planned
motion, then hands off to the goal's balancing controller without stopping the
simulation. Sparse libraries can route through intermediate configurations;
the panel shows the route, current edge and progress, or an explicit no-route
message. The picker is disabled during a request.

Share a target with `#quad/UUUD`: it starts hanging and then requests UUUD live.
A bare `#quad` stays all-down. The ghost chain uses the active controller's
absolute estimated angles and cart position. Angle and cart plots show errors
from the current target (the time-varying nominal during a transition). The
scene fits the full swing envelope.

Transitions are offline direct-collocation motions in
`data/transitions/<preset>/`. The exporter copies them to
`public/presets/<preset>/transitions/`, generating an index from completed files
when the library is still being built. Rerun the exporter as generation finishes
to include new routes. Only requested route files are fetched in the browser.

The tracker samples cubic Hermite trajectories, linearises the nonlinear chain
by central differences, uses Padé matrix exponentials for ZOH discretisation,
and designs a time-varying LQR with a backward Riccati pass. A time-varying
Kalman covariance update runs online. Design runs in batches budgeted to 4 ms,
yielding to the browser between batches while the balancing loop continues;
controllers are cached for subsequent requests. `SimStatus.designMs` and
`designMaxBatchMs` report wall time and the largest design batch.

Parity tests cover all-up and mixed balancing controllers, MuJoCo plant traces,
and the DD→UU tracker (nominal samples, discrete matrices, gains, Kalman states
and commands). Closed-loop tests cover one- and two-link swing-ups and the quad
swing-up when its source file exists. Equilibrium step/tap checks remain nominal
checks, not robustness guarantees.

## Deploying

The only artifact to deploy is `web/dist/`, produced by `npm run build`.
No special server headers or rewrites are needed.

### 0. Docker / Coolify

`web/Dockerfile` builds the site with Node 22 and serves `dist/` with nginx on
port 80 (`web/nginx.conf`). Test it locally:

```bash
docker build -t quad-pendulum web
docker run --rm -p 8080:80 quad-pendulum   # open http://localhost:8080
```

In Coolify, don't use the Nixpacks build pack: it has no start command for a
static site, and its Nix setup step can fail before your code is built. Set:

- **Build Pack:** Dockerfile
- **Base Directory:** `/web`
- **Dockerfile Location:** `/Dockerfile`
- **Ports Exposes:** `80`

### 1. Any static host

Run `npm run build` and upload the contents of `web/dist/` to any static file
host (S3, nginx, a plain web server, a folder on an existing site).

### 2. Netlify

`web/netlify.toml` sets the build command and publish directory.

- Base directory: `web`
- Build command: `npm run build`
- Publish directory: `dist`

Netlify reads `netlify.toml` from the base directory, so the file above is
picked up automatically.

### 3. Vercel

`web/vercel.json` sets the build and output settings.

- Root directory: `web`
- Framework preset: Vite
- Output directory: `dist`

### 4. GitHub Pages

Either publish `dist/` yourself (for example to a `gh-pages` branch), or use an
Actions workflow. Because `base` is `./`, the site works under the
`https://<user>.github.io/<repo>/` sub-path without changes. A minimal workflow
(create it at `.github/workflows/pages.yml` in the repo root):

```yaml
name: Deploy web
on:
  push:
    branches: [master]
permissions:
  contents: read
  pages: write
  id-token: write
concurrency:
  group: pages
  cancel-in-progress: true
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 22
          cache: npm
          cache-dependency-path: web/package-lock.json
      - run: npm ci
        working-directory: web
      - run: npm run build
        working-directory: web
      - uses: actions/upload-pages-artifact@v3
        with:
          path: web/dist
  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - id: deployment
        uses: actions/deploy-pages@v4
```

In the repo settings, set Pages > Build and deployment > Source to
"GitHub Actions".

### 5. Cloudflare Pages

- Root directory: `web`
- Build command: `npm run build`
- Build output directory: `dist`
