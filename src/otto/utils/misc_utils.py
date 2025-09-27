import hashlib
import os
import pathlib


def calculate_checksum(file_path: str) -> str:
        """Calculate SHA256 checksum of file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

def get_default_cache_location() -> str:
    """Returns a path to the default CACHE location, or $HOME/.cache."""
    cache_path = None
    if "CACHE" in os.environ and os.environ["CACHE"]:
        cache_path = os.environ["CACHE"]
    else:
        cache_path = str(pathlib.Path.home().joinpath(".cache"))

    # Check if the cache path exists, if not create it
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    return cache_path

def make_safe_filename(s):
    def safe_char(c):
        if c.isalnum():
            return c
        else:
            return "_"

    return "".join(safe_char(c) for c in s).rstrip("_")