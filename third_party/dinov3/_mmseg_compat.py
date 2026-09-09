# Copyright (c) Meta Platforms, Inc. and affiliates.
# Minimal compatibility shim for mmseg.models.builder.BACKBONES.
# DINOv3 vision_transformer.py uses @BACKBONES.register_module() as a
# class decorator.  This shim provides a no-op registry so the decorator
# works without installing the full mmseg package.


class _Registry:
    """Minimal registry that accepts register_module() as an identity decorator."""

    def register_module(self, *args, **kwargs):
        def decorator(cls):
            return cls
        return decorator


BACKBONES = _Registry()
