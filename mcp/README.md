# kokon-mcp

MCP server for [Kokon](https://kokon.app). It lets Claude Code, Cursor, Claude Desktop and other
MCP clients generate and edit images with Qwen-Image 2.1, running offline on your own GPU.
No API key, no cloud. Everything stays on `127.0.0.1`.

> **Requires the Kokon desktop app** (Windows with a 12 GB+ GPU: NVIDIA, or AMD/Intel in beta;
> or an Apple Silicon Mac, beta, 16 GB+). Download it from [kokon.app](https://kokon.app), open it once and
> finish the setup so the model is downloaded. This package is only the bridge between your AI
> agent and the app. The app doesn't need to be open: the server starts the engine on demand.

## Setup

Claude Code:

```bash
claude mcp add --scope user kokon -- npx -y kokon-mcp
```

Any client that takes the standard JSON config (`.mcp.json`, Claude Desktop, Cursor...):

```json
{
  "mcpServers": {
    "kokon": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "kokon-mcp"]
    }
  }
}
```

Needs Node.js 22 or newer. The Kokon installer also ships the same server as a standalone
executable (`kokon-mcp.exe` / `kokon-mcp` inside the app), if you prefer not to use Node. See the
[main README](https://github.com/GabrieleLozupone/kokon#use-it-from-your-ai-agent).

Then ask: *"Generate a hero image for this README and save it in docs/"* or *"Remove the background
clutter from screenshot.png"*. The first call takes about a minute while the model loads.

## Tools

| Tool | What it does |
| --- | --- |
| `generate_image` | Text to image. Size chosen automatically, or set width/height or aspect ratio. |
| `edit_image` | Edit an image with an instruction. Keeps its size; never overwrites the input. |
| `edit_images` | Create an image from up to 10 reference images ("image 1", "image 2"...). |
| `inpaint_image` | Change only a masked area: edit, remove, replace or add something. |
| `outpaint_image` | Extend an image beyond its borders. |
| `upscale_image` | Enlarge an image (default 2x, up to 8192 px). |
| `get_image_info` | Size, type and, for Kokon images, the prompt, seed and settings. |
| `get_generator_status` | Engine state, GPU, VRAM and job queue. |

## Behavior and safety

- Every result returns the absolute path of the saved file, its size, seed and metadata.
- Reads and writes are limited to the project folder (`CLAUDE_PROJECT_DIR` or the working
  directory), Kokon's output folder and folders listed in `QLS_MCP_ALLOWED_DIRS` (separated by
  semicolons). Existing files are never overwritten.
- Images are not sent back to the model by default. Pass `include_preview: true` for a small JPEG.
- Progress is reported through MCP progress notifications.

## License

AGPL-3.0-only. The Qwen-Image 2.1 model weights the app downloads are under the Qwen Research
License (non-commercial). Source: [github.com/GabrieleLozupone/kokon](https://github.com/GabrieleLozupone/kokon/tree/master/mcp).
