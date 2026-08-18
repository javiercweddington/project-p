from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        name="project-p",
        version="0.1.0",
        description="A Python project",
        author="JavierCW",
        author_email="servingthroughscience [at] gmail [dot] com",
        license="MIT",
        packages=find_packages(),
        python_requires=">=3.8",
        install_requires=[
            "requests>=2.31.0",
            "numpy>=1.24.0",
            "pandas>=2.0.0",
        ],
        extras_require={
            "dev": [
                "pytest>=7.4.0",
                "black>=23.0.0",
                "flake8>=6.0.0",
            ],
        },
    )
