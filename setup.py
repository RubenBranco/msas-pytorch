from setuptools import setup
import re
from pathlib import Path


def get_version() -> str:
    with open(Path(__file__).parent / "msas_pytorch" / "__init__.py") as f:
        for line in f:
            if line.startswith("__version__"):
                return re.split(r"['\"]", line)[1]


setup(
    name="msas-pytorch",
    version=get_version(),
    description="Unofficial implementation of 'Multi-Sequence Aggregate Similarity' (MSAS) in PyTorch",
    author="Ruben Branco",
    author_email="rmbranco@fc.ul.pt",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
    ],
    keywords="generative modeling, time series, pytorch, metrics",
    packages=[
        "msas_pytorch",
    ],
    install_requires=[
        "torch",
        "scipy",
    ],
    url="https://github.com/RubenBranco/msas-pytorch",
)
