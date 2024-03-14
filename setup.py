from setuptools import setup, find_packages

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.MD").read_text()

setup(
    name="mimm",
    description="MLX Image Models",
    author="Robert McCraith",
    author_email="mccraithrobert@gmail.com",
    packages=find_packages(),
    install_requires=[
        "mlx"
    ],
    long_description=long_description,
    long_description_content_type='text/markdown',
)
