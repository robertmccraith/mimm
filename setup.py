from setuptools import setup, find_packages


setup(
    name="mimm",
    description="MLX Image Models",
    author="Robert McCraith",
    author_email="mccraithrobert@gmail.com",
    packages=find_packages(),
    install_requires=[
        "mlx"
    ],
    long_description=open("README.md").read(),
    long_description_content_type='text/markdown',
)
