import path from "node:path";
import { describe, expect, it } from "vitest";
import { localDataBase } from "../src/service.js";

// Must match the service's dirs::data_local_dir(), or the MCP server misses service.json.
describe("data folder", () => {
  it("follows each system's local data folder", () => {
    expect(localDataBase("win32", { LOCALAPPDATA: "C:\Users\me\AppData\Local" }, "C:\Users\me")).toBe("C:\Users\me\AppData\Local");
    expect(localDataBase("darwin", {}, "/Users/me")).toBe(path.join("/Users/me", "Library", "Application Support"));
    expect(localDataBase("linux", {}, "/home/me")).toBe(path.join("/home/me", ".local", "share"));
    expect(localDataBase("linux", { XDG_DATA_HOME: "/data/xdg" }, "/home/me")).toBe("/data/xdg");
    expect(localDataBase("linux", { XDG_DATA_HOME: "relative" }, "/home/me")).toBe(path.join("/home/me", ".local", "share"));
  });
});
