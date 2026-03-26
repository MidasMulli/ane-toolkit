from setuptools import setup, find_packages

setup(
    name="ane-toolkit",
    version="0.1.0",
    description="Deploy custom activation functions to Apple Neural Engine",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Nick L.",
    url="https://github.com/MidasMulli/ane-toolkit",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=["numpy", "torch"],
    extras_require={"coreml": ["coremltools"]},
    license="MIT",
)
