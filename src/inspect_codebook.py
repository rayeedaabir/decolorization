"""Audit the codebook against the claims the paper makes about it.

§2.4 of the draft says: "All 16 prototypes are optimal for some image, indicating the
codebook is neither redundant nor under-sized." That is a checkable claim, and the live
demo raised a doubt about it -- the selected prototype P15 printed weights
(0.22, 0.71, 0.07, 0.00, 0.00), which is BT.709 (0.2126, 0.7152, 0.0722, 0, 0) to within
0.008. If two prototypes are near-duplicates then a codebook slot is wasted and the
sentence is vulnerable.

    python inspect_codebook.py --codebook codebook.npz

Reports: the weights of every prototype, how often each is the per-image best, and every
pair closer together than --dup-tol. Fix the paper's wording to whatever this prints.
"""
import argparse
import numpy as np
from cues import CUE_NAMES, BT709


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codebook", default="codebook.npz")
    ap.add_argument("--dup-tol", type=float, default=0.05,
                    help="L1 distance below which two prototypes are near-duplicates")
    a = ap.parse_args()

    d = np.load(a.codebook, allow_pickle=True)
    P, E = d["protos"], d["E"]
    K, n = len(P), len(E)
    best = E.argmax(1)
    counts = np.bincount(best, minlength=K)

    print(f"{K} prototypes, {n} training images\n")
    print(f"{'':4s}" + "".join(f"{c:>9s}" for c in CUE_NAMES)
          + f"{'best for':>10s}{'share':>8s}{'L1 to BT.709':>14s}")
    print("-" * 76)
    for j in range(K):
        l1 = float(np.abs(P[j] - BT709).sum())
        flag = "  <- BT.709" if l1 < 1e-6 else ("  <- ~BT.709" if l1 < a.dup_tol else "")
        print(f"P{j:<3d}" + "".join(f"{v:9.4f}" for v in P[j])
              + f"{counts[j]:10d}{counts[j]/n*100:7.1f}%{l1:14.4f}{flag}")

    unused = [j for j in range(K) if counts[j] == 0]
    print(f"\nprototypes that are best for at least one image: {K - len(unused)}/{K}")
    if unused:
        print(f"NEVER the best for any image: {', '.join(f'P{j}' for j in unused)}")
        print("  -> the paper cannot say 'all 16 prototypes are optimal for some image'.")

    print(f"\nnear-duplicate pairs (L1 < {a.dup_tol}):")
    dup = [(i, j, float(np.abs(P[i] - P[j]).sum()))
           for i in range(K) for j in range(i + 1, K)
           if np.abs(P[i] - P[j]).sum() < a.dup_tol]
    if not dup:
        print("  none - the codebook is genuinely diverse, the claim stands as written.")
    else:
        for i, j, v in sorted(dup, key=lambda x: x[2]):
            print(f"  P{i} and P{j}: L1 = {v:.4f}   "
                  f"(best for {counts[i]} and {counts[j]} images)")
        print("  -> soften the sentence: say how many DISTINCT strategies the codebook")
        print("     holds, or note that the k-means centroid nearest the standard")
        print("     duplicates the appended BT.709 prototype. Do not claim no redundancy.")

    # how much would dropping the duplicates cost?
    if dup:
        drop = sorted({j for _, j, _ in dup})
        keep = [j for j in range(K) if j not in drop]
        loss = (E.max(1) - E[:, keep].max(1)).mean()
        print(f"\ndropping {len(drop)} duplicate(s) would cost {loss:.5f} mean E "
              f"({loss / E.max(1).mean() * 100:.3f}%) -- if that is negligible, say so;")
        print("it turns a weakness into evidence that the codebook is not over-sized.")


if __name__ == "__main__":
    main()
