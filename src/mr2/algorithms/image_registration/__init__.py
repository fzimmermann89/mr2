"""Image registration."""

from mr2.algorithms.image_registration.affine_registration import affine_registration
from mr2.algorithms.image_registration.correlation_registration import correlation_registration
from mr2.algorithms.image_registration.register_images import register_images
from mr2.algorithms.image_registration.spline_registration import spline_registration

__all__ = ["affine_registration", "correlation_registration", "register_images", "spline_registration"]
