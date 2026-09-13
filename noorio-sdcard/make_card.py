#!/usr/bin/env python3
"""Build the SD card that turns a stock T41 camera into an SSH dev box.

  python3 sdcard/make_card.py [-o OUTDIR]

Asks for a root password, a wifi SSID and its passphrase, then writes
`OUTDIR/factorytest/` containing the setup script with those values already
computed, plus the dropbear binary it installs.

The values are precomputed here because the camera cannot derive them itself:
at the point this script runs there is no wpa_passphrase, no openssl, and no
python on the device -- only busybox.  So the SSID goes in as hex, the wifi PSK
as PBKDF2-SHA1, and the root password as MD5-crypt.

See README.md for what to do with the result.
"""
import argparse
import getpass
import hashlib
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "factorytest.sh.in")
DROPBEAR = os.path.join(HERE, "dropbear")

_B64 = "./0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def _to64(value, n):
    out = ""
    for _ in range(n):
        out += _B64[value & 0x3F]
        value >>= 6
    return out


def md5_crypt(password: bytes, salt: bytes) -> str:
    """MD5-crypt (`$1$`), the hash this camera's /etc/shadow uses.

    Implemented here rather than with the `crypt` module, which is deprecated
    and removed in Python 3.13 -- this tool should outlive that.  Algorithm per
    the original Poul-Henning Kamp implementation; `main()` self-tests it
    against a known pair before using it on anything.
    """
    salt = salt[:8]
    ctx = hashlib.md5(password + b"$1$" + salt)

    alt = hashlib.md5(password + salt + password).digest()
    n = len(password)
    ctx.update(alt * (n // 16) + alt[:n % 16])

    # the length, as a trail of either NULs or the password's first byte
    i = n
    while i:
        ctx.update(b"\0" if i & 1 else password[:1])
        i >>= 1
    digest = ctx.digest()

    # 1000 rounds, deliberately serial
    for i in range(1000):
        c = hashlib.md5()
        c.update(password if i & 1 else digest)
        if i % 3:
            c.update(salt)
        if i % 7:
            c.update(password)
        c.update(digest if i & 1 else password)
        digest = c.digest()

    out = ""
    for a, b, c in ((0, 6, 12), (1, 7, 13), (2, 8, 14), (3, 9, 15), (4, 10, 5)):
        out += _to64((digest[a] << 16) | (digest[b] << 8) | digest[c], 4)
    out += _to64(digest[11], 2)
    return f"$1${salt.decode()}${out}"


def _self_test():
    """A wrong hash locks you out of a camera with no serial console, so prove
    the implementation on a known pair before trusting it with a real one."""
    known = "$1$WSCbquzY$rbZNWA2Z1E6tXWf.Drqsr."
    got = md5_crypt(b"admin", b"WSCbquzY")
    if got != known:
        sys.exit(f"md5_crypt self-test FAILED\n  expected {known}\n  got      {got}")


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", default=os.path.join(HERE, "out"),
                    help="where to write factorytest/ (default: sdcard/out)")
    args = ap.parse_args(argv[1:])

    _self_test()

    host = input("hostname [pluto-cam]: ").strip() or "pluto-cam"
    ssid = input("wifi SSID: ").strip()
    if not ssid:
        sys.exit("a wifi SSID is required -- the camera has no other way back to you")
    wifi_pw = getpass.getpass("wifi passphrase: ")
    if len(wifi_pw) < 8:
        sys.exit("WPA2 passphrases are at least 8 characters")
    root_pw = getpass.getpass("root password for the camera: ")
    if not root_pw:
        sys.exit("an empty root password would leave dropbear refusing every login")
    if root_pw != getpass.getpass("root password again: "):
        sys.exit("the two root passwords differ")

    salt = "".join(_B64[b & 0x3F] for b in os.urandom(8)).encode()
    values = {
        "HOSTNAME": host,
        "WIFI_SSID_HEX": ssid.encode().hex(),
        "WIFI_PSK_HEX": hashlib.pbkdf2_hmac(
            "sha1", wifi_pw.encode(), ssid.encode(), 4096, 32).hex(),
        "ROOT_HASH": md5_crypt(root_pw.encode(), salt),
    }

    script = open(TEMPLATE).read()
    for key, value in values.items():
        script = script.replace(f"@@{key}@@", value)
    left = [tok for tok in ("@@",) if tok in script]
    if left:
        sys.exit("template still has unfilled placeholders")

    out = os.path.join(args.out, "factorytest")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "factorytest.sh")
    with open(path, "w") as f:
        f.write(script)
    os.chmod(path, 0o755)

    # A syntax error here is only discoverable by watching a camera fail to boot.
    if subprocess.run(["sh", "-n", path]).returncode:
        sys.exit(f"{path} is not valid shell -- not shipping it")

    if os.path.exists(DROPBEAR):
        shutil.copy2(DROPBEAR, out)
        os.chmod(os.path.join(out, "dropbear"), 0o755)
    else:
        print(f"WARNING: {DROPBEAR} is missing -- the card will not install SSH")

    print(f"\nwrote {out}/")
    for name in sorted(os.listdir(out)):
        print(f"  {name}  ({os.path.getsize(os.path.join(out, name))} bytes)")
    print(f"\nCopy that factorytest/ directory to the root of a FAT-formatted SD\n"
          f"card, put it in the camera, and power on.  After ~90 s:\n"
          f"    ssh root@{host}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
