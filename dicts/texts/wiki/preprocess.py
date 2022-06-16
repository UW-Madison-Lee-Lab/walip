data = "wiki"
srclang = "en"
tgtlang = "ja"
path = f"orig_{data}_{srclang}_{tgtlang}_test.txt"

srcset = []
tgtset = []
with open(path, "r") as f:
    for line in f:
        src, tgt = line.rstrip().split()
        if src not in srcset:
            srcset.append(src)
        tgtset.append(tgt)

with open(f"{data}_{srclang}_{tgtlang}_{srclang}_test.txt", "w") as f:
    for word in srcset:
        f.write(word + "\n")

with open(f"{data}_{srclang}_{tgtlang}_{tgtlang}_test.txt", "w") as f:
    for word in tgtset:
        f.write(word + "\n")


