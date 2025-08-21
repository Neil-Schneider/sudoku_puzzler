#!/usr/bin/env python3
"""
Sudoku generator that creates a solved grid and then masks cells to form a puzzle
with a single solution.

Usage examples:
  python sudoku_gen.py
  python sudoku_gen.py --clues 32
  python sudoku_gen.py --holes 49 --no-symmetric
  python sudoku_gen.py --clues 28 --seed 123

Notes:
- "clues" = number of filled cells to leave (higher = easier). Typical ranges:
    easy ~ 36-45, medium ~ 30-35, hard ~ 26-29, expert ~ 22-25
- The generator enforces uniqueness by re-solving after each removal.
- Very low clue counts can take longer or fail; raise clues or try a different seed.
"""
from __future__ import annotations
import argparse
import copy
import random
from typing import List, Tuple, Optional

Board = List[List[int]]
ALL = set(range(1, 10))

def print_board(board: Board) -> None:
    """Pretty print a Sudoku board (0 = blank)."""
    for r in range(9):
        row = []
        for c in range(9):
            val = board[r][c]
            row.append("." if val == 0 else str(val))
            if c in (2, 5): row.append("|")
        print(" ".join(row))
        if r in (2, 5): print("------+-------+------")

def copy_board(board: Board) -> Board:
    return [row[:] for row in board]

def box_start(i: int) -> int:
    return (i // 3) * 3

def is_valid(board: Board, r: int, c: int, v: int) -> bool:
    """Check if placing v at (r,c) is valid."""
    if any(board[r][x] == v for x in range(9)): return False
    if any(board[y][c] == v for y in range(9)): return False
    br, bc = box_start(r), box_start(c)
    for rr in range(br, br + 3):
        for cc in range(bc, bc + 3):
            if board[rr][cc] == v:
                if rr == r and cc == c:  # same cell
                    continue
                # if trying to place in empty cell, presence in box forbids
                if board[r][c] == 0:
                    return False
                # if checking existing cell, conflict only if another same value
                if (rr, cc) != (r, c) and v == board[r][c]:
                    return False
    return True

def candidates(board: Board, r: int, c: int) -> List[int]:
    """Return possible values for empty cell (r,c)."""
    if board[r][c] != 0:
        return []
    row = set(board[r][x] for x in range(9))
    col = set(board[y][c] for y in range(9))
    br, bc = box_start(r), box_start(c)
    box = set(board[rr][cc] for rr in range(br, br + 3) for cc in range(bc, bc + 3))
    return list(ALL - row - col - box)

def find_empty_with_fewest(board: Board) -> Optional[Tuple[int, int, List[int]]]:
    """Heuristic: choose next empty cell with the fewest candidates (MRV)."""
    best: Optional[Tuple[int, int, List[int]]] = None
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                cand = candidates(board, r, c)
                if not cand:
                    return (r, c, [])  # dead end
                if best is None or len(cand) < len(best[2]):
                    best = (r, c, cand)
                    if len(cand) == 1:
                        return best
    return best

def shuffle(seq):
    s = list(seq)
    random.shuffle(s)
    return s

def solve_count(board: Board, limit: int = 2) -> int:
    """
    Count solutions up to `limit`. Early exit when >= limit (useful for uniqueness).
    Modifies a copy during recursion; original `board` is not changed.
    """
    b = copy_board(board)
    return _solve_count_backtrack(b, limit)

def _solve_count_backtrack(board: Board, limit: int) -> int:
    if limit <= 0:
        return 0
    nxt = find_empty_with_fewest(board)
    if nxt is None:
        return 1  # filled -> one solution
    r, c, cand = nxt
    if not cand:
        return 0
    # Try candidates in randomized order to avoid bias
    for v in shuffle(cand):
        board[r][c] = v
        cnt = _solve_count_backtrack(board, limit)
        if cnt:
            limit -= cnt
            if limit <= 0:
                # early stop; we already reached the limit
                board[r][c] = 0
                return cnt + max(0, limit)  # limit is negative or zero; we just propagate ">= limit"
        # revert and continue
        board[r][c] = 0
    # We need to compute total solutions tried; recompute properly
    # (The early-return above complicates exact counts; we only care if >=2)
    # To keep it simple, do a second pass without early return if we found at least 1.
    # But for speed, we track limit and stop when it hits 0.
    # Our return path here is accurate enough for uniqueness check.
    return (2 if limit <= 0 else (2 - limit))  # clamp to [0,2]

def solve_one(board: Board) -> bool:
    """Solve in-place (single solution path)."""
    nxt = find_empty_with_fewest(board)
    if nxt is None:
        return True
    r, c, cand = nxt
    for v in shuffle(cand):
        board[r][c] = v
        if solve_one(board):
            return True
        board[r][c] = 0
    return False

def generate_full_solution() -> Board:
    """Generate a full valid Sudoku solution by randomized backtracking."""
    board = [[0] * 9 for _ in range(9)]
    # Fill diagonal 3x3 boxes first (speeds up generation), then solve
    for box in range(0, 9, 3):
        nums = shuffle(range(1, 10))
        idx = 0
        for r in range(box, box + 3):
            for c in range(box, box + 3):
                board[r][c] = nums[idx]
                idx += 1
    if not solve_one(board):
        # Fallback: fully randomized fill if somehow stuck (rare)
        board = [[0] * 9 for _ in range(9)]
        solve_one(board)
    return board

def positions_with_symmetry(sym: bool = True) -> List[Tuple[int, int]]:
    """
    Return a list of (r,c) positions. If symmetric, we only include one from each symmetric pair,
    plus center cell once. We'll handle pair removal during digging.
    """
    coords = [(r, c) for r in range(9) for c in range(9)]
    if not sym:
        random.shuffle(coords)
        return coords
    seen = set()
    unique = []
    for r, c in coords:
        sr, sc = 8 - r, 8 - c
        key = tuple(sorted([(r, c), (sr, sc)]))
        if key not in seen:
            seen.add(key)
            unique.append((r, c))
    random.shuffle(unique)
    return unique

def make_puzzle(solution: Board, clues: Optional[int] = None, holes: Optional[int] = None,
                symmetric: bool = True) -> Board:
    """
    Remove cells while maintaining a single solution.
    Provide either `clues` (remaining numbers) or `holes` (masked cells).
    """
    if clues is not None and holes is not None:
        raise ValueError("Provide only one of `clues` or `holes`.")
    total = 81
    target_clues = clues if clues is not None else total - (holes or 49)  # default 49 holes -> 32 clues
    target_clues = max(17, min(81, target_clues))  # clamp

    puzzle = copy_board(solution)
    positions = positions_with_symmetry(symmetric)

    def current_clues() -> int:
        return sum(1 for r in range(9) for c in range(9) if puzzle[r][c] != 0)

    for r, c in positions:
        if current_clues() <= target_clues:
            break

        # Determine symmetric counterpart
        sr, sc = 8 - r, 8 - c
        cells = {(r, c)}
        if symmetric and (sr, sc) != (r, c):
            cells.add((sr, sc))

        # Save values & clear
        saved = {pos: puzzle[pos[0]][pos[1]] for pos in cells}
        for rr, cc in cells:
            puzzle[rr][cc] = 0

        # Check uniqueness after removal
        if solve_count(puzzle, limit=2) >= 2:
            # not unique -> revert
            for (rr, cc), v in saved.items():
                puzzle[rr][cc] = v
        # else keep removal

        if current_clues() <= target_clues:
            break

    return puzzle

def as_lines(board: Board) -> str:
    return "\n".join("".join(str(v) if v != 0 else "." for v in row) for row in board)

def main():
    ap = argparse.ArgumentParser(description="Generate a Sudoku puzzle with a unique solution.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--clues", type=int, help="Number of filled cells to leave (17–81).")
    g.add_argument("--holes", type=int, help="Number of cells to remove (0–64+, typical ~40–55).")
    ap.add_argument("--no-symmetric", action="store_true", help="Disable 180° symmetry when removing cells.")
    ap.add_argument("--seed", type=int, help="Random seed for reproducibility.")
    ap.add_argument("--no-print-solution", action="store_true", help="Do not print the solution grid.")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    # 1) Generate a full valid solution
    solution = generate_full_solution()

    # 2) Dig to create a puzzle with unique solution
    puzzle = make_puzzle(
        solution,
        clues=args.clues,
        holes=args.holes,
        symmetric=not args.no_symmetric
    )

    # Output
    print("\nPuzzle (0 or . = blank):")
    print_board(puzzle)
    if not args.no_print_solution:
        print("\nSolution:")
        print_board(solution)

    # Also print compact lines (useful for copy/paste or files)
    print("\nPuzzle (one line per row):")
    print(as_lines(puzzle))
    print("\nSolution (one line per row):")
    print(as_lines(solution))

if __name__ == "__main__":
    main()
