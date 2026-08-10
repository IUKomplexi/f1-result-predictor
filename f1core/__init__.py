"""Core shared modules for the F1 result predictor.

``predict``, ``config``, ``reporting`` and ``httpclient`` live here so every
package (``f1data``, ``f1weather``, ``features``, ``model``, ``f1web``) imports
them from a single, installable package instead of via ``sys.path`` hacks.
"""
