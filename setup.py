from setuptools import setup, find_packages

setup(
    name="nlpchat",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "sentence-transformers",
        "scikit-learn",
        "numpy",
        "pickle-mixin"
    ],
    author="AKM Korishee Apurbo",
    description="An easy-to-use chatbot creation package with NLP intent identification.",
    url="https://github.com/IMApurbo/nlpchat",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
