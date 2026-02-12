"""Custom setuptools-scm version scheme for branch-aware versioning.

Computes PEP 440 versions based on the current git branch:
  - ``main`` (tagged): exact tag version (e.g. ``2.3.1``)
  - ``dev``: major bump + beta (e.g. ``3.0.0b5``)
  - ``feat/*``: minor bump + alpha (e.g. ``2.4.0a3``)
  - ``fix/*``, ``bug/*``: patch bump + rc (e.g. ``2.3.2rc2``)
  - other: patch bump + dev (e.g. ``2.3.2.dev4``)

The prerelease number is the commit distance from the last tag,
which naturally prevents collisions between parallel branches.
"""


def scheme(version):
    """Compute a PEP 440 version string from git state.

    Args:
        version: ``setuptools_scm.ScmVersion`` instance carrying the
            latest tag, commit distance, node hash, and branch name.

    Returns:
        str: A PEP 440 version string.
    """
    tag = version.tag
    distance = version.distance
    branch = version.branch or ""

    if distance is None or distance == 0:
        return str(tag)

    # tag is a packaging.version.Version; .release gives (major, minor, patch)
    major, minor, patch = tag.release[:3]

    if branch.startswith("feat/"):
        return f"{major}.{minor + 1}.0a{distance}"
    elif branch.startswith(("fix/", "bug/")):
        return f"{major}.{minor}.{patch + 1}rc{distance}"
    elif branch == "dev":
        return f"{major + 1}.0.0b{distance}"
    else:
        return f"{major}.{minor}.{patch + 1}.dev{distance}"
