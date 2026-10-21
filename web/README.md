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

## Deploying

The only artifact to deploy is `web/dist/`, produced by `npm run build`.
No special server headers or rewrites are needed.

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
