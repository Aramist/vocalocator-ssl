from setuptools import setup

setup(
    name="contrastive",
    version="1.0.0",
    description="Tool for sound-source localization of rodent vocalizations",
    url="https://github.com/neurostatslab/gerbilizer",
    author="NeuroStats Lab",
    license="MIT",
    packages=["contrastive", "contrastive.utils"],
    install_requires=["h5py", "numpy", "torch", "scikit-learn"],
)
