"""
VerifAI Command Line Interface
===============================

Provides command-line access to VerifAI detection capabilities.

Usage:
    $ verifai detect image.jpg
    $ verifai detect video.mp4 --output results.json
    $ verifai info
"""

from cli.main import cli

__all__ = ["cli"]
