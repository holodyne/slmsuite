# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import base64
import os
import re
import sys
import inspect
import shutil
import types

import requests


# import numpydoc

module_paths = [
    os.path.abspath(""),
    os.path.abspath("../.."),
    os.path.abspath("../../slmsuite"),
]
for module_path in module_paths:
    sys.path.insert(0, module_path)


from examples import download_example_notebooks

# Without a system pandoc, nbsphinx uses the one pypandoc_binary bundles.
if shutil.which("pandoc") is None:
    try:
        import pypandoc
        os.environ["PATH"] = os.path.dirname(pypandoc.get_pandoc_path()) + os.pathsep + os.environ["PATH"]
    except (ImportError, OSError):
        pass

# -- Project information -----------------------------------------------------

project = "slmsuite"
copyright = "2021-2025 slmsuite Developers. 2026 Holodyne Labs, Inc."
author = "Holodyne Labs, Inc."
# Read the version from the package, as pyproject.toml does, so it never needs a manual bump.
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "slmsuite", "__init__.py")) as f:
    release = re.search(r"""^__version__ = ['"]([^'"]+)['"]""", f.read(), re.M).group(1)

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx_design",
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "nbsphinx",
    "sphinx_copybutton",
    "sphinx_last_updated_by_git",
    # "sphinxcontrib.video",
]

extlinks = {
    "issue": ("https://github.com/holodyne/slmsuite/issues/%s", "GH"),
    "pull": ("https://github.com/holodyne/slmsuite/pull/%s", "PR"),
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "cupy": ("https://docs.cupy.dev/en/stable", None),
    "torch": ("https://docs.pytorch.org/docs/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}

# Adapted from https://github.com/DisnakeDev/disnake/blob/7853da70b13fcd2978c39c0b7efa59b34d298186/docs/conf.py#L192
def linkcode_resolve(domain, info):
    if domain != 'py':
        return None

    try:
        obj = sys.modules[info["module"]]
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)
        obj = inspect.unwrap(obj)

        if isinstance(obj, property):
            obj = inspect.unwrap(obj.fget)

        path = os.path.relpath(inspect.getsourcefile(obj))
        path = path.split('slmsuite/')[-1]
        src, lineno = inspect.getsourcelines(obj)
    except Exception:
        return None

    path = f"{path}#L{lineno}-L{lineno + len(src) - 1}"
    return f"https://github.com/holodyne/slmsuite/blob/main/slmsuite/" + path

# Add any paths that contain templates here, relative to this directory.
templates_path = ["templates"]

# numpydoc_xref_param_type = True
# numpydoc_xref_ignore = {"optional", "type_without_description", "BadException"}
# # Run docstring validation as part of build process
# numpydoc_validation_checks = {"all", "GL01", "SA04", "RT03"}
toc_object_entries_show_parents = 'hide'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True
autosummary_ignore_module_all = False  # Respect __all__ (e.g. toolbox.phase re-exports).
autodoc_member_order = "bysource"   # This doesn't work for autosummary unfortunately
                                    # https://github.com/sphinx-doc/sphinx/issues/5379
# autodoc_typehints = "signature"
napoleon_use_param = True
add_module_names = False # Remove namespaces from class/method signatures

nbsphinx_execute = "never"
nbsphinx_allow_errors = True #continue through jupyter errors

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# copybutton_prompt_text = "$ "

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

# html_theme = "sphinx_rtd_theme"
navigation_with_keys = True

html_sidebars = {
    "**": [],
}
# html_sidebars = {
#     "**": ["sidebar-nav-bs", "sidebar-ethical-ads"]
# }
html_context = {
    "default_mode": "auto",
}

html_title = f"{project} v{release} Manual"
html_last_updated_fmt = "%b %d, %Y"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the built-in "default.css".
# html_static_path = [] # ["static"]
# html_static_path = ['static']
# html_css_files = ['custom.css']

# Add a logo
# html_theme_options = {"logo_only": True}
html_logo = "static/slmsuite.svg"

# Add a favicon
html_favicon = "static/slmsuite-notext-32x32.ico"

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "logo": {
        "image_light": "static/slmsuite.svg",
        "image_dark": "static/slmsuite-dark.svg",
    },
    "show_prev_next": True,
    "navbar_end": ["theme-switcher", "navbar-icon-links"], #, "search-field.html"
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/holodyne/slmsuite/",
            "icon": "fab fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/slmsuite/",
            "icon": "fab fa-python",
        },
    ],
    "footer_start": ["copyright", "sphinx-version"],
    "footer_end": ["last-updated", "theme-version"],
    # "content_footer_items": ["last-updated"],
    "show_version_warning_banner": True,
    "navbar_center": ["navbar-nav"],   # "version-switcher",  # https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/version-dropdown.html
    # "switcher": {
    #     "version_match": switcher_version,
    #     "json_url": "https://numpy.org/doc/_static/versions.json",
    # },
    "secondary_sidebar_items": {
        "**" : ["page-toc", "edit-this-page", "sourcelink"],
        "index" : [],
        "examples" : [],
    }, # , "breadcrumbs"
    "show_toc_level": 4
    # "secondary_sidebar_end": ["sidebar-ethical-ads"],
}

# https://stackoverflow.com/questions/5599254/how-to-use-sphinxs-autodoc-to-document-a-classs-init-self-method
def skip(app, what, name, obj, would_skip, options):
    skip_ = would_skip
    # Document `__init__`.
    if name in ("__init__",):
        skip_ = False
    # Don't document magic things.
    elif name in ("__dict__", "__doc__", "__weakref__", "__module__"):
        skip_ = True
    # Don't document private things.
    elif name[0] == '_':
        skip_ = True
    # Don't document members inherited from C builtins (e.g. ``int.to_bytes`` on an ``IntEnum``).
    elif isinstance(obj, (
        types.BuiltinFunctionType, types.MethodDescriptorType, types.WrapperDescriptorType,
        types.GetSetDescriptorType, types.MemberDescriptorType, types.ClassMethodDescriptorType,
    )):
        skip_ = True

    return skip_

def public_bases(app, name, obj, options, bases):
    """
    Show each base class as its nearest public ancestor, by its public path (e.g.
    ``algorithms.FeedbackHologram`` rather than ``algorithms._feedback.FeedbackHologram``).
    """
    import importlib

    def public(cls):
        if cls is object:
            return []
        module = ".".join(p for p in cls.__module__.split(".") if not p.startswith("_"))
        try:
            twin = getattr(importlib.import_module(module), cls.__name__, None)
        except ImportError:
            twin = None
        if (
            cls.__name__.startswith("_") or not isinstance(twin, type)
            or twin is obj or not issubclass(twin, cls)
        ):
            return [p for base in cls.__bases__ for p in public(base)]
        return [cls if twin is cls and module == cls.__module__ else f":class:`~{module}.{cls.__name__}`"]

    bases[:] = list(dict.fromkeys(p for base in bases for p in public(base))) or [object]

def resolve_relative(app, env, node, contnode):
    """
    Resolve relative references to slmsuite objects that Sphinx misses. Docstrings are
    written from the perspective of their own class, so ``:attr:`aperture``` in an
    inherited method documented on a subclass page should link to the base class's
    attribute. Tries, in order: the context class's MRO, then a unique match among the
    documented objects. References to private names render as plain code.
    """
    import importlib
    from sphinx.util.nodes import make_refnode

    if node.get("refdomain") != "py":
        return None
    target = node["reftarget"].lstrip("~").lstrip(".")
    objects = env.get_domain("py").objects

    def unique(names):
        """Collapse aliases (e.g. ``toolbox.phase._zernike.ZernikeBasis``) of one object."""
        anchors = {}
        for name in names:
            anchors.setdefault((objects[name].docname, objects[name].node_id), name)
        return list(anchors.values())

    def link(name):
        entry = objects[name]
        return make_refnode(app.builder, node["refdoc"], entry.docname, entry.node_id, contnode, name)

    # 1) Walk the MRO of the class this docstring is documented under.
    module, cls = node.get("py:module"), node.get("py:class")
    if module and cls:
        try:
            obj = importlib.import_module(module)
            for part in cls.split("."):
                obj = getattr(obj, part)
            for base in getattr(obj, "__mro__", ()):
                # The documented (public) path of this base, e.g. ``algorithms._hologram``
                # is documented as ``algorithms``.
                public = ".".join(p for p in base.__module__.split(".") if not p.startswith("_"))
                name = f"{public}.{base.__name__}.{target}"
                if name in objects:
                    return link(name)
        except (ImportError, AttributeError):
            pass

    # 2) A unique module member (``func``, ``Class.attr``); not :mod:, e.g. the external pylablib.
    if node.get("reftype") != "mod":
        modules = env.get_domain("py").modules
        matches = unique([
            n for n in objects
            if n.startswith("slmsuite") and n.endswith("." + target)
            and n[:-len(target) - 1] in modules
        ])
        if len(matches) == 1:
            return link(matches[0])

    # 3) Private names are undocumented by design: show them as code, without a link.
    if target.split(".")[-1].startswith("_"):
        return contnode

    return None

# relative to this directory
examples_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_examples")
images_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "_build", "html", "_images"
)

def setup(app):
    app.connect("autodoc-skip-member", skip)
    app.connect("autodoc-process-bases", public_bases)
    app.connect("missing-reference", resolve_relative)

    # Use local notebooks
    # examples_source = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..", "slmsuite-examples/examples")
    # shutil.copytree(examples_source,examples_path)

    download_example_notebooks(
        examples_path=examples_path,
        images_path=images_path,
    )