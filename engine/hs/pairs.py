"""Which image pairs the solve matches — pure functions, no pycolmap (hs solve --estimate and
the tests import this without COLMAP; rigcolmap.py and monocolmap.py import it to write the
pair list they hand to ``pycolmap.match_image_pairs``).

Two matchers:

``exhaustive``   every image against every other: n·(n−1)/2 pairs. What the vendored scripts
                 always did, and still their default. Right when a view is revisited from far
                 apart in time and nothing else would pair the two visits.

``sequential``   rig-aware and deterministic, built from captures.json's order:
                 * window — capture i against captures i+1 … i+overlap, every eye against every
                   eye (L_i–L_j, L_i–R_j, R_i–L_j, R_i–R_j on a stereo rig);
                 * rig mates — L_i–R_i, always;
                 * loop pass — every ``loop_stride``-th capture (0, s, 2s, …) against every other
                   stride capture, all eyes, so the ends of an orbit and a walk-around that
                   comes back meet each other without a vocabulary tree.
                 Linear in captures apart from the loop pass, which is (n/s)²/2 · eyes².

COLMAP's own SequentialPairingOptions is not used for the window: it orders images by name,
so over the L/ and R/ folders it pairs L/cap421 with R/cap000 and never pairs a capture's two
eyes (checked on pycolmap 4.2.0: 7 captures, overlap 2 → 25 pairs, none of them L_i–R_i).

``resolve_matcher("auto", n)`` is sequential above ``AUTO_SEQUENTIAL_ABOVE`` captures, else
exhaustive: below that exhaustive costs minutes and is the safer choice.
"""
import os

MATCHERS = ("exhaustive", "sequential")
MATCHER_CHOICES = ("exhaustive", "sequential", "auto")
AUTO_SEQUENTIAL_ABOVE = 60
DEFAULT_OVERLAP = 15
DEFAULT_LOOP_STRIDE = 8
IMPORTED_BLOCK_SIZE = 1225      # pycolmap 4.2.0 ImportedPairingOptions.block_size default


def resolve_matcher(matcher, n_captures, threshold=AUTO_SEQUENTIAL_ABOVE):
    """'auto' -> 'sequential' when captures > threshold, else 'exhaustive'. Others pass through."""
    if matcher == "auto":
        return "sequential" if n_captures > threshold else "exhaustive"
    if matcher not in MATCHERS:
        raise ValueError(f"unknown matcher {matcher!r} (want one of {', '.join(MATCHER_CHOICES)})")
    return matcher


def image_name(eye, capture, ext=".jpg"):
    """The database name COLMAP gives an image under images/: 'L/cap000.jpg'."""
    return f"{eye}/{capture}{ext}"


def _canon(a, b):
    return (a, b) if a < b else (b, a)


def sequential_pairs(captures, eyes=("L", "R"), overlap=DEFAULT_OVERLAP,
                     loop_stride=DEFAULT_LOOP_STRIDE, ext=".jpg", loop=True):
    """The sequential pair list over ``captures`` (capture ids in capture order, e.g.
    ['cap000', 'cap001', …]). Returns a sorted list of unique (name_a, name_b) with
    name_a < name_b. ``loop=False`` leaves the stride pass out (the vocab-tree loop replaces it).
    ``loop_stride`` ≤ 0 also disables it."""
    caps = list(captures)
    n = len(caps)
    overlap = max(0, int(overlap))
    out = set()

    def add_caps(i, j):
        for ea in eyes:
            for eb in eyes:
                out.add(_canon(image_name(ea, caps[i], ext), image_name(eb, caps[j], ext)))

    for i in range(n):
        for k in range(len(eyes)):                      # rig mates
            for m in range(k + 1, len(eyes)):
                out.add(_canon(image_name(eyes[k], caps[i], ext), image_name(eyes[m], caps[i], ext)))
        for j in range(i + 1, min(n, i + overlap + 1)):  # window
            add_caps(i, j)
    if loop and loop_stride and loop_stride > 0:
        stride = list(range(0, n, int(loop_stride)))
        for a in range(len(stride)):
            for b in range(a + 1, len(stride)):
                add_caps(stride[a], stride[b])
    return sorted(out)


def exhaustive_pair_count(n_images):
    return n_images * (n_images - 1) // 2


def pair_count(matcher, n_captures, n_eyes=2, overlap=DEFAULT_OVERLAP, loop_stride=DEFAULT_LOOP_STRIDE):
    """Pairs a matcher will run for n_captures captures of n_eyes images each. Sequential is
    counted by building the list (exact, and 422 captures takes milliseconds)."""
    m = resolve_matcher(matcher, n_captures)
    if m == "exhaustive":
        return exhaustive_pair_count(n_captures * n_eyes)
    eyes = ("L", "R")[:n_eyes] if n_eyes <= 2 else tuple(f"E{k}" for k in range(n_eyes))
    caps = [f"c{i:06d}" for i in range(n_captures)]
    return len(sequential_pairs(caps, eyes=eyes, overlap=overlap, loop_stride=loop_stride))


def write_pair_list(pairs, path):
    """COLMAP's match-list format: one 'name_a name_b' per line."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for a, b in pairs:
            f.write(f"{a} {b}\n")
    return path


def exhaustive_block_progress(i, ni, j, nj):
    """Progress for COLMAP's 'Processing block [i/ni, j/nj]' line, which is printed as a block
    starts: (blocks finished, blocks in all). pycolmap 4.2.0 visits EVERY ni×nj block — each
    holds about half its block²'s pairs, so together they partition the n(n−1)/2 pairs (checked
    on synthetic images: 14 images, block 4 → all 16 blocks logged, [2/4, 1/4] included)."""
    return (i - 1) * nj + (j - 1), ni * nj


def exhaustive_blocks(n_images, block_size=50):
    nb = -(-n_images // block_size)
    return nb * nb


def match_sequential(db, work, caps, eyes, a, fm=None):
    """Run the sequential matcher on a COLMAP database (rigcolmap.py and monocolmap.py call
    this; pycolmap is imported here, not at the top, so the rest of the module stays pure).

    Writes ``<work>/match_pairs.txt`` and feeds it to ``pycolmap.match_image_pairs``, which
    logs 'Processing block [k/n]' per IMPORTED_BLOCK_SIZE pairs. ``a`` carries overlap,
    loop_stride, loop ('stride' | 'vocab'), vocab_tree, vocab_neighbors. With ``loop='vocab'``
    the stride pass is left out and ``pycolmap.match_vocabtree`` retrieves loop candidates
    instead; COLMAP skips pairs the window already matched."""
    import sys
    import pycolmap
    vocab = getattr(a, "loop", "stride") == "vocab"
    tree = (getattr(a, "vocab_tree", None) or os.environ.get("HS_VOCAB_TREE")) if vocab else None
    if vocab and (not tree or not os.path.exists(tree)):
        sys.exit(f"--loop vocab needs a vocabulary tree (--vocab-tree / HS_VOCAB_TREE): {tree!r} "
                 f"not found: `hs tools --fetch-vocab-tree`, or use --loop stride")
    plist = sequential_pairs(caps, eyes=eyes, overlap=a.overlap, loop_stride=a.loop_stride, loop=not vocab)
    path = write_pair_list(plist, os.path.join(work, "match_pairs.txt"))
    n_img = len(caps) * len(eyes)
    loop = ("vocab-tree loop pass" if vocab else
            f"loop stride {a.loop_stride}" if a.loop_stride and a.loop_stride > 0 else "no loop pass")
    print(f"sequential matching {n_img} images ({len(plist)} pairs; overlap {a.overlap}, {loop})...", flush=True)
    po = pycolmap.ImportedPairingOptions()
    po.match_list_path = path
    kw = {"matching_options": fm} if fm is not None else {}
    pycolmap.match_image_pairs(db, pairing_options=po, **kw)
    if vocab:
        vo = pycolmap.VocabTreePairingOptions()
        vo.vocab_tree_path = tree
        vo.num_images = int(a.vocab_neighbors)
        print(f"vocab-tree loop pass: {a.vocab_neighbors} nearest images each ({tree})...", flush=True)
        pycolmap.match_vocabtree(db, pairing_options=vo, **kw)
    return len(plist)
