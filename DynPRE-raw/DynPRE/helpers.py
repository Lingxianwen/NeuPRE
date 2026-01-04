# -*- coding: utf-8 -*
import difflib
import errno
import os
import shutil


def hexdump(src: bytes, length=16, sep=".") -> str:
    """Hex dump bytes to ASCII string, padded neatly
    In [107]: x = b'\x01\x02\x03\x04AAAAAAAAAAAAAAAAAAAAAAAAAABBBBBBBBBBBBBBBBBBBBBBBBBB'

    In [108]: print('\n'.join(hexdump(x)))
    00000000  01 02 03 04 41 41 41 41  41 41 41 41 41 41 41 41 |....AAAAAAAAAAAA|
    00000010  41 41 41 41 41 41 41 41  41 41 41 41 41 41 42 42 |AAAAAAAAAAAAAABB|
    00000020  42 42 42 42 42 42 42 42  42 42 42 42 42 42 42 42 |BBBBBBBBBBBBBBBB|
    00000030  42 42 42 42 42 42 42 42                          |BBBBBBBB        |
    """
    FILTER = "".join([(len(repr(chr(x))) == 3) and chr(x) or sep for x in range(256)])
    lines = []
    for c in range(0, len(src), length):
        chars = src[c : c + length]
        hex_ = " ".join(["{:02x}".format(x) for x in chars])
        if len(hex_) > 24:
            hex_ = "{} {}".format(hex_[:24], hex_[24:])
        printable = "".join(
            ["{}".format((x <= 127 and FILTER[x]) or sep) for x in chars]
        )
        lines.append(
            "{0:08x}  {1:{2}s} |{3:{4}s}|".format(
                c, hex_, length * 3, printable, length
            )
        )
        print(printable)
    return lines


def highlight_diff(hex_str1, hex_str2):
    d = difflib.Differ()
    diff = list(d.compare(hex_str1, hex_str2))

    result = []
    for line in diff:
        if line.startswith("+"):
            result.append(line)
        elif line.startswith("-"):
            result.append(line)
        else:
            result.append(line)

    return "".join(result)


def mkdir_safe(directory_name: str) -> None:
    try:
        os.makedirs(directory_name)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def rm_safe(path: str):
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)


def compare_hex_strings(hex_str1, hex_str2):
    d = difflib.Differ()
    diff = list(d.compare(hex_str1, hex_str2))

    print("Diff between hex strings:")
    for line in diff:
        if line.startswith("+") or line.startswith("-"):
            print(line)


def ascii_dump(src: bytes, sep=".") -> str:
    FILTER = "".join([(len(repr(chr(x))) == 3) and chr(x) or sep for x in range(256)])
    printable = "".join(["{}".format((x <= 127 and FILTER[x]) or sep) for x in src])
    return printable


def hexstream_to_bytes(hexstream):
    return bytes.fromhex(hexstream)
