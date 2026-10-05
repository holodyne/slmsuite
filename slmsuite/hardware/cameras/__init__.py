# The first line of the docstring might read awkward alone,
# but it's meant to parallel the slms module description.
"""
The sensor arrays used to measure results.
Computer vision hardware is connected to python by a myriad of SDKs, often provided by hardware
vendors. However, these SDKs likewise have a myriad of function names and hardware-specific
quirks. Thus, cameras in :mod:`slmsuite` are integrated as subclasses of
the abstract class :class:`.Camera`, which requires subclasses to implement a number of methods
relevant for SLM feedback (see below).
These subclasses are effectively wrappers for the given SDK, but also include
quality-of-life features such as image transformations (flips, rotates) and useful common methods.

Tip
~~~
While the superclass :class:`.Camera` only requires a small number of features to
be implemented as class functions, any further control of a camera interface can be
accessed by using the given SDK object directly (usually the attribute ``cam`` of
the subclass) or writing additional functions into the subclass.
"""
