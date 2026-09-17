"""rig.npz view layout — one place that knows whether the views are stereo pairs or single
cameras, so the path builders, coverage and the view scorer do not each assume ``[::2]``.

Two layouts share the file (rigcolmap.py / monocolmap.py ``write_rig_npz``):

  stereo (Hydrogen clips)  names interleaved ``capNNN_L, capNNN_R, …``; a *capture* is a pair
                            and the left eye is the reference view: ``L = arange(0, n, 2)``.
  mono   (camera arrays)   one view per camera, ``names = GA_L, GB_L, …`` (the ``_L`` suffix
                            keeps ``images/L/<cam>.jpg`` and every ``name[:-2]`` image lookup
                            working); ``stereo = False`` in the npz and ``L = arange(n)``.

A file without a ``stereo`` key is stereo — every rig.npz written before arrays existed.
"""
import numpy as np


def is_stereo(G):
    return bool(G["stereo"]) if "stereo" in G.files else True


def left_indices(G, names=None):
    """Indices into names/K/R/t/C of the reference view of each capture."""
    n = len(names) if names is not None else len(G["names"])
    return np.arange(0, n, 2) if is_stereo(G) else np.arange(n)


def right_index(G, left):
    """The right eye's index for a reference view, or None on a mono rig."""
    return int(left) + 1 if is_stereo(G) else None


def load(rig_npz):
    """-> (G, names, L): the npz, its view names and the reference-view indices."""
    G = np.load(rig_npz, allow_pickle=True)
    names = [str(x) for x in G["names"]]
    return G, names, left_indices(G, names)


def capture_name(name):
    """``cap004_L`` -> ``cap004``, ``GA_L`` -> ``GA``: the image stem the view came from."""
    return name[:-2] if name.endswith(("_L", "_R")) else name
