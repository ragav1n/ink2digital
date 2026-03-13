"""
Setup file for German Lecture Slide OCR project.
Allows importing project modules from anywhere with: pip install -e .
"""
from setuptools import setup, find_packages

setup(
    name='german_lecture_ocr',
    version='0.1.0',
    packages=find_packages(exclude=['venv', 'data', 'outputs', 'runs', 'checkpoints']),
    install_requires=[],  # See requirements_project.txt
)
