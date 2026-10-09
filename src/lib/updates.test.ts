import { describe, expect, it } from "vitest";
import { autoCheckEnabled, checkDue } from "./updates";

describe("update check schedule", () => {
  it("is on unless turned off", () => {
    expect(autoCheckEnabled(undefined)).toBe(true);
    expect(autoCheckEnabled({})).toBe(true);
    expect(autoCheckEnabled({ updateCheck: true })).toBe(true);
    expect(autoCheckEnabled({ updateCheck: false })).toBe(false);
  });
  it("checks at most once a day", () => {
    const now = Date.UTC(2026, 9, 10, 12);
    expect(checkDue(null, now)).toBe(true);
    expect(checkDue(NaN, now)).toBe(true);
    expect(checkDue(now - 60_000, now)).toBe(false);
    expect(checkDue(now - 25 * 3600_000, now)).toBe(true);
  });
});
