import zstandard as zstd
import sys
your_filename = "lichess_db_atomic_rated_2023-02.pgn.zst"
with open(your_filename, "rb") as f:
    data = f.read()
#print(data)
dctx = zstd.ZstdDecompressor()
decompressed = dctx.decompress(data)
'''
import pandas as pd
df = pd.read(decompressed)
print(df.head())
'''
print(decompressed)
print(len(decompressed))
print(sys.getsizeof(decompressed))
f = open("ac_pgns.txt", "w")
#f.write(decompressed)
f.close()