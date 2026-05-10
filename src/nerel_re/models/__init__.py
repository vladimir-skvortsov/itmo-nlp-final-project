"""Relation extraction model architectures."""

from nerel_re.models.baseline import BaselineREModel
from nerel_re.models.typed_markers import TypedMarkersREModel

__all__ = ['BaselineREModel', 'TypedMarkersREModel']
