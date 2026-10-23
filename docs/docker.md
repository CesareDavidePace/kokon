# Kokon in Docker (NVIDIA)

The same Kokon as the Linux packages, without a window: you use it from a browser on any device.
For NVIDIA graphics cards (CUDA). An image for AMD and Intel cards (Vulkan) will follow once it can
be tested.

## What you need

- Linux with an NVIDIA card and the proprietary driver (`nvidia-smi` works). Windows works too through
  Docker Desktop with WSL 2.
- Docker with the Compose plugin, and the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- About 15 GB of disk for the engine and the model.

## Start it

Each release publishes the image as `ghcr.io/gabrielelozupone/kokon`, so you need only the
compose file, no clone and no build:

```bash
mkdir kokon && cd kokon
curl -fsSLO https://raw.githubusercontent.com/GabrieleLozupone/kokon/master/docker/compose.yaml
docker compose up -d
docker exec -it kokon kokon-service --set-password   # once; at least 8 characters
```

Open `http://<this computer's address>:39152` in a browser, sign in, and follow the setup: it
downloads the engine and the model (about 12 GB) into the container's data volume, once.

## Where things are

| Volume | Inside the container | What |
|---|---|---|
| `kokon-data` | `/data` | engine, models, gallery database, logs |
| `kokon-config` | `/config` | settings |
| `kokon-gallery` | `/gallery` | the images saved to the gallery |

To keep the saved images in a folder of your own, replace the `kokon-gallery` line in
`compose.yaml` with a folder Kokon can write to (it runs as user id 1000):

```bash
mkdir -p ~/kokon-gallery && sudo chown 1000:1000 ~/kokon-gallery
#   - ~/kokon-gallery:/gallery
```

## Everyday use

```bash
docker logs -f kokon                                  # what the service is doing
docker exec -it kokon kokon-service --set-password    # a new password (every browser signs in again)
docker compose down                                   # stop (the volumes stay)
docker compose pull && docker compose up -d           # update to a new version
```

The `latest` tag follows every release. To get only the fixes of one version, write for example
`ghcr.io/gabrielelozupone/kokon:0.11` in `compose.yaml` (it stays on 0.11.x); each release also
has its exact tag, such as `:0.11.0`.

## Build the image yourself

From a clone of this repository (takes a while: it compiles the UI and `kokon-service`):

```bash
docker build -f docker/Dockerfile -t ghcr.io/gabrielelozupone/kokon:latest .
docker compose -f docker/compose.yaml up -d
```

## Security

- Nothing is reachable until a password is set. The password is stored as a hash, and 5 wrong tries
  in a row make that device wait.
- The port is published on every network interface. Docker's published ports bypass host firewalls
  such as ufw; to use Kokon only from the computer itself, change the port line to
  `"127.0.0.1:39152:39152"`.
- The connection is plain HTTP: use it on your home network, and through Tailscale or another VPN from
  anywhere else. Never forward the port on your router. See [remote-access.md](remote-access.md).
- Kokon's local API with its token (used by scripts and the MCP server) stays inside the container.
