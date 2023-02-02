__all__ = ['burst','processing.xspectra.compute_IW_subswath_intraburst_xspectra',
           'processing.xspectra.compute_IW_subswath_interburst_xspectra','tools']
try:
    from importlib import metadata
except ImportError: # for Python<3.8
    import importlib_metadata as metadata
__version__ = metadata.version('xsarslc')