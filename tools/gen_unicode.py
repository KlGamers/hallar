import sys
import unicodedata
import urllib.request

MAXCP = 0x110000


def ranges(pred):
    out, start = [], None
    for cp in range(MAXCP):
        hit = pred(cp)
        if hit and start is None:
            start = cp
        elif not hit and start is not None:
            out.append((start, cp - 1))
            start = None
    if start is not None:
        out.append((start, MAXCP - 1))
    return out


def cat(cp):
    return unicodedata.category(chr(cp))


def is_ws(cp):
    return cp in (9, 10, 11, 12, 13, 0x85) or cat(cp) in ("Zs", "Zl", "Zp")


def is_ctrl(cp):
    if chr(cp) in "\t\n\r":
        return False
    return cat(cp).startswith("C")


def is_punct(cp):
    if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126:
        return True
    return cat(cp).startswith("P")


def is_mark(cp):
    return cat(cp) == "Mn"


def lower_map():
    out = []
    for cp in range(MAXCP):
        ch = chr(cp)
        lo = ch.lower()
        if lo != ch:
            cps = [ord(c) for c in lo]
            if len(cps) > 3:
                raise SystemExit(f"lowercase of U+{cp:04X} longer than 3")
            out.append((cp, cps + [0] * (3 - len(cps))))
    return out


def nfd_map():
    out = []
    for cp in range(MAXCP):
        ch = chr(cp)
        d = unicodedata.normalize("NFD", ch)
        if d != ch:
            cps = [ord(c) for c in d]
            if len(cps) > 4:
                raise SystemExit(f"NFD of U+{cp:04X} longer than 4")
            out.append((cp, cps + [0] * (4 - len(cps))))
    return out


GB = ["Other", "CR", "LF", "Control", "Extend", "ZWJ", "Regional_Indicator", "Prepend", "SpacingMark",
      "L", "V", "T", "LV", "LVT"]


def ucd(name):
    ver = unicodedata.unidata_version
    url = f"https://www.unicode.org/Public/{ver}/ucd/{name}"
    with urllib.request.urlopen(url) as res:
        return res.read().decode()


def parse(text, want=None):
    out = []
    for ln in text.splitlines():
        ln = ln.split("#", 1)[0].strip()
        if not ln:
            continue
        rng, prop = [x.strip() for x in ln.split(";")[:2]]
        if want and prop != want:
            continue
        a, _, b = rng.partition("..")
        out.append((int(a, 16), int(b or a, 16), prop))
    return sorted(out)


def emit_gb(fh):
    rows = parse(ucd("auxiliary/GraphemeBreakProperty.txt"))
    fh.write("constexpr Prop kGrapheme[] = {\n")
    for a, b, prop in rows:
        fh.write(f"    {{0x{a:X}, 0x{b:X}, {GB.index(prop)}}},\n")
    fh.write("};\n\n")
    pict = parse(ucd("emoji/emoji-data.txt"), "Extended_Pictographic")
    emit(fh, "Pict", [(a, b) for a, b, _ in pict])


def emit(fh, name, rng):
    fh.write(f"constexpr Range k{name}[] = {{\n")
    for a, b in rng:
        fh.write(f"    {{0x{a:X}, 0x{b:X}}},\n")
    fh.write("};\n\n")


def main(path):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f'constexpr const char* kUnicodeVersion = "{unicodedata.unidata_version}";\n\n')
        emit(fh, "Space", ranges(is_ws))
        emit(fh, "Control", ranges(is_ctrl))
        emit(fh, "Punct", ranges(is_punct))
        emit(fh, "Mark", ranges(is_mark))
        emit_gb(fh)
        fh.write("constexpr Map3 kLower[] = {\n")
        for cp, cps in lower_map():
            fh.write(f"    {{0x{cp:X}, {{0x{cps[0]:X}, 0x{cps[1]:X}, 0x{cps[2]:X}}}}},\n")
        fh.write("};\n\nconstexpr Map4 kNfd[] = {\n")
        for cp, cps in nfd_map():
            fh.write(f"    {{0x{cp:X}, {{0x{cps[0]:X}, 0x{cps[1]:X}, 0x{cps[2]:X}, 0x{cps[3]:X}}}}},\n")
        fh.write("};\n")


if __name__ == "__main__":
    main(sys.argv[1])
