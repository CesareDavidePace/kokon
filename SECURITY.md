# Security policy

## Reporting a vulnerability

Please report security problems privately through GitHub's private vulnerability reporting:
open the repository's **Security** tab and click **Report a vulnerability**
(https://github.com/GabrieleLozupone/kokon/security/advisories/new). Don't open a public
issue for them.

Include what you found, how to reproduce it, and the Kokon version and OS. Kokon is maintained
by one person: expect a first reply within about a week. Fixes ship through the in-app updater,
and you'll be credited in the advisory unless you prefer not to be.

## Supported versions

Only the latest release gets security fixes.

## What Kokon exposes

Useful context for reports:

- **Network.** Kokon makes two kinds of outgoing request: during setup, downloads from
  `github.com` (the stable-diffusion.cpp release) and `huggingface.co` (the models); and, at
  most once a day unless turned off, a fetch of `latest.json` from this repository's latest
  GitHub release to check for a new version. Update installers are downloaded only when the user
  clicks **Update**, and installed only if their signature matches the public key built into the
  app. Every model and engine download is checked against a pinned SHA-256
  (`crates/qls-core/src/catalog.rs`). Nothing about the user or their images is sent.
- **Local API.** `kokon-service` listens on `127.0.0.1` only (port 39151 by default) and needs
  a bearer token from `KokonData/service.json` for every call except `/ping`. Requests that carry a browser `Origin`
  header are refused. `sd-server` listens on `127.0.0.1:39150`.
- **MCP server.** `kokon-mcp` runs over stdio. Reads and writes are limited to the project
  folder, Kokon's output folder and folders in `QLS_MCP_ALLOWED_DIRS`, and existing files are
  never overwritten.

Examples of things worth reporting: a way to reach the API from a web page or another user
account, a path that escapes the allowed folders, an update or download that is accepted
without a valid signature or checksum, or data leaving the machine that the README says doesn't.

Problems in stable-diffusion.cpp, Tauri or the model weights themselves are best reported to
those projects; if they affect Kokon, a heads-up here is still welcome.
