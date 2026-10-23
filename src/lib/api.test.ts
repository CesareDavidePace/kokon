// The client as a browser uses it (vitest runs without the Tauri shell, like a browser on
// another device): no token anywhere, the CSRF header on every change, and a sign-in again
// when the session ends.

import { afterEach, describe, expect, it, vi } from "vitest";
import { api, configure, setSignedOutHandler, urls } from "./api";
import { inBrowser } from "./platform";

function reply(status: number, body: unknown) {
  return Promise.resolve(new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } }));
}

afterEach(() => vi.unstubAllGlobals());

describe("api in a browser", () => {
  it("runs as a browser in tests", () => expect(inBrowser).toBe(true));

  it("loads images and events without a token", () => {
    configure("/api/v1", "");
    expect(urls.gen("abc")).toBe("/api/v1/gallery/abc/image");
    expect(urls.inputThumb("x")).toBe("/api/v1/inputs/x/thumb");
    expect(urls.lutPreview("warm", "gen:1", 512)).toBe("/api/v1/luts/warm/preview?source=gen%3A1&max=512");
  });

  it("sends the CSRF header on changes only, never a bearer token", async () => {
    configure("/api/v1", "");
    const fetch = vi.fn(() => reply(200, { ok: true }));
    vi.stubGlobal("fetch", fetch);
    await api.status();
    await api.setFavorite("g1", true);
    const [get, patch] = fetch.mock.calls as unknown as [string, RequestInit][];
    expect(get[0]).toBe("/api/v1/status");
    expect(get[1].headers).not.toHaveProperty("Authorization");
    expect(get[1].headers).not.toHaveProperty("x-kokon-csrf");
    expect(patch[1].headers).toMatchObject({ "x-kokon-csrf": "1", "Content-Type": "application/json" });
    expect(patch[1].headers).not.toHaveProperty("Authorization");
  });

  it("asks to sign in again when the session ends, but not for a wrong password", async () => {
    configure("/api/v1", "");
    const signedOut = vi.fn();
    setSignedOutHandler(signedOut);
    const ended = { error: { code: "login_required", message: "Sign in with the password." } };
    vi.stubGlobal("fetch", vi.fn(() => reply(401, ended)));
    await expect(api.gallery({})).rejects.toThrow("Sign in");
    expect(signedOut).toHaveBeenCalledTimes(1);
    await expect(api.login("wrong one")).rejects.toThrow();
    expect(signedOut).toHaveBeenCalledTimes(1);
  });
});
