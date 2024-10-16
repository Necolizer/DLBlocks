# from https://www.quickprogrammingtips.com/python/how-to-calculate-md5-hash-of-a-file-in-python.html

import hashlib

filename = r''
md5_hash = hashlib.md5()
with open(filename,"rb") as f:
    # Read and update hash in chunks of 4K
    for byte_block in iter(lambda: f.read(4096),b""):
        md5_hash.update(byte_block)
    print(md5_hash.hexdigest())
