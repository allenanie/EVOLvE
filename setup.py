import os
import setuptools

here = os.path.abspath(os.path.dirname(__file__))
version = {}
with open(os.path.join(here, "banditbench/version.py"), encoding="utf8") as fp:
    exec(fp.read(), version)
__version__ = version["__version__"]

install_requires = [
    "pydantic==2.10.5",
    "litellm",
    "pandas",
    "tensorflow_datasets==4.9.3",
    "tensorflow==2.17.0",
    "ml-dtypes==0.3.1",
    "matplotlib",
    "scipy"
]

setuptools.setup(
    name="banditbench",
    version=__version__,
    author="Allen Nie",
    author_email="anie@cs.stanford.edu",
    url="https://github.com/allenanie/EVOLvE",
    license='MIT LICENSE',
    description="BanditBench: A Bandit Benchmark to Evaluate Self-Improving LLM Algorithms",
    long_description=open('README.md', encoding="utf8").read(),
    packages=setuptools.find_packages(include=["banditbench*"]),
    install_requires=install_requires,
    python_requires=">=3.9",
)
