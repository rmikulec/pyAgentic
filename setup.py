"""Build-time setup for setuptools-scm version resolution.

All project metadata lives in pyproject.toml. This file exists solely
because setuptools-scm requires a callable (not an entry-point string)
for custom version schemes, and pyproject.toml cannot express that.

Version scheme (branch → version mapping):
  - main (tagged):  exact tag  (e.g. 2.3.1)
  - dev:            major + beta  (e.g. 3.0.0b5)
  - feat/*:         minor + alpha (e.g. 2.4.0a3)
  - fix/*|bug/*:    patch + rc    (e.g. 2.3.2rc2)
  - other:          patch + dev   (e.g. 2.3.2.dev4)
"""

from setuptools import setup


def _version_scheme(version):
    """Compute a PEP 440 version from git state."""
    tag = version.tag
    distance = version.distance
    branch = version.branch or ""

    if distance is None or distance == 0:
        return str(tag)

    # tag is a packaging.version.Version; .release gives (major, minor[, patch])
    release = tag.release
    major = release[0] if len(release) > 0 else 0
    minor = release[1] if len(release) > 1 else 0
    patch = release[2] if len(release) > 2 else 0

    if branch.startswith("feat/"):
        return f"{major}.{minor + 1}.0a{distance}"
    elif branch.startswith(("fix/", "bug/")):
        return f"{major}.{minor}.{patch + 1}rc{distance}"
    elif branch == "dev":
        return f"{major + 1}.0.0b{distance}"
    else:
        return f"{major}.{minor}.{patch + 1}.dev{distance}"


setup(
    use_scm_version={
        "version_scheme": _version_scheme,
        "local_scheme": "no-local-version",
    },
)
