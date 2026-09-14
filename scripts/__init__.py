"""Operational scripts.

A package rather than loose files so modules can share helpers by import
(`from scripts.prepare_artifacts import find_latest_run`) without the same file
being resolvable under two module names. Each script remains directly
executable.
"""
