from setuptools import setup, find_packages

setup(
    name="pypurr",
    version="0.1",
    packages=find_packages(),
    author="Matthieu Le Goff",
    author_email="matth.le.goff@gmail.com",
    description="Simple implementation of the VJ algorithm",
    long_description=open("README.md").read(),
    include_package_data=True,
    url="https://github.com/Matt3164/viola-jones-experiments",
    license="WTFPL",
    entry_points={
        "console_scripts": [
            "pypurr_cli = pypurr.cli.main:cli"
        ]
    }
)