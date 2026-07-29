import hashlib

def hash_to_int(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:8], 16)
