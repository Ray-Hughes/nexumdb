import { describe, expect, it } from "vitest";
import { diffChunks } from "../diff";

describe("diffChunks", () => {
  it("reports no change for identical versions", () => {
    const chunks = ["alpha", "beta", "gamma"];
    const result = diffChunks(chunks, chunks);
    expect(result.added).toBe(0);
    expect(result.removed).toBe(0);
    expect(result.unchanged).toBe(3);
    expect(result.lines.every((l) => l.op === "same")).toBe(true);
  });

  it("detects an inserted chunk without disturbing its neighbours", () => {
    const result = diffChunks(["a", "b"], ["a", "new", "b"]);
    expect(result.added).toBe(1);
    expect(result.removed).toBe(0);
    expect(result.unchanged).toBe(2);
    expect(result.lines.map((l) => l.op)).toEqual(["same", "added", "same"]);
  });

  it("detects a removed chunk", () => {
    const result = diffChunks(["a", "gone", "b"], ["a", "b"]);
    expect(result.removed).toBe(1);
    expect(result.added).toBe(0);
    expect(result.lines.find((l) => l.op === "removed")?.text).toBe("gone");
  });

  it("reports a replaced chunk as one removal and one addition", () => {
    const result = diffChunks(["a", "old", "c"], ["a", "new", "c"]);
    expect(result.added).toBe(1);
    expect(result.removed).toBe(1);
    expect(result.unchanged).toBe(2);
  });

  it("handles a completely rewritten document", () => {
    const result = diffChunks(["a", "b"], ["x", "y"]);
    expect(result.unchanged).toBe(0);
    expect(result.added).toBe(2);
    expect(result.removed).toBe(2);
  });

  it("handles empty versions in both directions", () => {
    expect(diffChunks([], ["a"]).added).toBe(1);
    expect(diffChunks(["a"], []).removed).toBe(1);
    expect(diffChunks([], []).lines).toHaveLength(0);
  });

  it("finds the longest common subsequence rather than aligning positionally", () => {
    // A prepended chunk shifts everything; a positional diff would call all
    // four lines changed, which would make the view useless.
    const result = diffChunks(["b", "c", "d"], ["a", "b", "c", "d"]);
    expect(result.unchanged).toBe(3);
    expect(result.added).toBe(1);
    expect(result.removed).toBe(0);
  });

  it("degrades to wholesale replacement rather than freezing on huge inputs", () => {
    const before = Array.from({ length: 2_100 }, (_, i) => `chunk ${i}`);
    const after = Array.from({ length: 2_100 }, (_, i) => `chunk ${i}`);
    const result = diffChunks(before, after);
    expect(result.added).toBe(2_100);
    expect(result.removed).toBe(2_100);
    expect(result.unchanged).toBe(0);
  });

  it("preserves every line from both versions", () => {
    const before = ["a", "b", "c"];
    const after = ["a", "x", "c", "d"];
    const result = diffChunks(before, after);
    const kept = result.lines.filter((l) => l.op !== "added").map((l) => l.text);
    const produced = result.lines.filter((l) => l.op !== "removed").map((l) => l.text);
    expect(kept).toEqual(before);
    expect(produced).toEqual(after);
  });
});
