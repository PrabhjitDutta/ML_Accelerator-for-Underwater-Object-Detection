import numpy as np, os, sys
a, b = sys.argv[1], sys.argv[2]
worst = 1.0
for f in sorted(os.listdir(a), key=lambda s: (len(s), s)):
    x = np.fromfile(os.path.join(a, f), np.float32).astype(np.float64)
    y = np.fromfile(os.path.join(b, f), np.float32).astype(np.float64)
    assert x.size == y.size, f
    c = x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-30)
    worst = min(worst, c)
    print("%-16s n=%9d cos=%.6f maxabs=%.4g" % (f, x.size, c, np.abs(x - y).max()))
print("WORST cos %.6f over %d dumps" % (worst, len(os.listdir(a))))
sys.exit(0 if worst >= 0.999 else 1)
