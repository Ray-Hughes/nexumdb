/**
 * Chunk-level diff between two document versions.
 *
 * Documents are immutable and re-ingestion writes a whole new set of chunks,
 * so "what changed" is a sequence alignment over chunk texts rather than a
 * text diff. Longest-common-subsequence gives the alignment; anything not on
 * it is an insertion or a deletion.
 */

export type DiffOp = "same" | "added" | "removed";

export interface DiffLine {
  op: DiffOp;
  text: string;
  /** Index in the version the line came from. */
  index: number;
}

export interface DiffSummary {
  lines: DiffLine[];
  added: number;
  removed: number;
  unchanged: number;
}

/**
 * Align two sequences of chunk texts.
 *
 * The LCS table is O(n*m); chunk counts per document are in the hundreds at
 * most, so this stays trivial. A cap guards the pathological case rather than
 * letting the UI freeze.
 */
export function diffChunks(before: string[], after: string[]): DiffSummary {
  const CAP = 2_000;
  if (before.length > CAP || after.length > CAP) {
    // Too large to align meaningfully in the UI; report wholesale replacement
    // rather than pretending to a line-level answer.
    return {
      lines: [
        ...before.map((text, index) => ({ op: "removed" as const, text, index })),
        ...after.map((text, index) => ({ op: "added" as const, text, index })),
      ],
      added: after.length,
      removed: before.length,
      unchanged: 0,
    };
  }

  const rows = before.length;
  const cols = after.length;
  const table: number[][] = Array.from({ length: rows + 1 }, () =>
    new Array<number>(cols + 1).fill(0),
  );

  for (let i = rows - 1; i >= 0; i--) {
    for (let j = cols - 1; j >= 0; j--) {
      table[i]![j]! =
        before[i] === after[j]
          ? table[i + 1]![j + 1]! + 1
          : Math.max(table[i + 1]![j]!, table[i]![j + 1]!);
    }
  }

  const lines: DiffLine[] = [];
  let added = 0;
  let removed = 0;
  let unchanged = 0;
  let i = 0;
  let j = 0;

  while (i < rows && j < cols) {
    if (before[i] === after[j]) {
      lines.push({ op: "same", text: before[i]!, index: j });
      unchanged++;
      i++;
      j++;
    } else if (table[i + 1]![j]! >= table[i]![j + 1]!) {
      lines.push({ op: "removed", text: before[i]!, index: i });
      removed++;
      i++;
    } else {
      lines.push({ op: "added", text: after[j]!, index: j });
      added++;
      j++;
    }
  }
  while (i < rows) {
    lines.push({ op: "removed", text: before[i]!, index: i });
    removed++;
    i++;
  }
  while (j < cols) {
    lines.push({ op: "added", text: after[j]!, index: j });
    added++;
    j++;
  }

  return { lines, added, removed, unchanged };
}
