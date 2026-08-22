import { describe, expect, it } from "vitest";
import { formatBytes, shortId, truncate } from "../format";

describe("shortId", () => {
  it("uses the random tail of a UUIDv7, not the timestamp prefix", () => {
    // These two differ only after the timestamp portion, which is exactly the
    // case that made an 8-char prefix useless.
    const a = "01a02b85-8cc7-7070-9313-cdecdd25d543";
    const b = "01a02b85-8cc7-7071-8000-aaaabbbbcccc";
    expect(shortId(a)).not.toBe(shortId(b));
    expect(shortId(a)).toHaveLength(8);
    expect(shortId(a)).toBe("dd25d543");
  });
});

describe("formatBytes", () => {
  it("scales through units", () => {
    expect(formatBytes(0)).toBe("0 B");
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(2048)).toBe("2.0 KB");
    expect(formatBytes(5 * 1024 * 1024)).toBe("5.0 MB");
  });
});

describe("truncate", () => {
  it("collapses whitespace and marks cuts", () => {
    expect(truncate("hello   world", 20)).toBe("hello world");
    expect(truncate("hello world", 8)).toBe("hello w…");
    expect(truncate("exact", 5)).toBe("exact");
  });

  it("strips newlines that would break single-line layout", () => {
    expect(truncate("line\none", 40)).toBe("line one");
  });
});
