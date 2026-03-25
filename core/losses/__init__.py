def build_loss_modules(*args, **kwargs):
    from .builder import build_loss_modules as _build_loss_modules

    return _build_loss_modules(*args, **kwargs)



def compute_loss_modules(*args, **kwargs):
    from .builder import compute_loss_modules as _compute_loss_modules

    return _compute_loss_modules(*args, **kwargs)



def required_aux_heads(*args, **kwargs):
    from .builder import required_aux_heads as _required_aux_heads

    return _required_aux_heads(*args, **kwargs)



def requires_depth_render(*args, **kwargs):
    from .builder import requires_depth_render as _requires_depth_render

    return _requires_depth_render(*args, **kwargs)


def requires_geom_depth_render(*args, **kwargs):
    from .builder import requires_geom_depth_render as _requires_geom_depth_render

    return _requires_geom_depth_render(*args, **kwargs)


__all__ = [
    "build_loss_modules",
    "compute_loss_modules",
    "required_aux_heads",
    "requires_depth_render",
    "requires_geom_depth_render",
]
