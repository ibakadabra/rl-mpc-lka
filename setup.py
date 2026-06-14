from setuptools import setup, find_packages

setup(
    name="rl-mpc-autonomous-vehicle",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "cvxpy>=1.4",
        "osqp>=0.6",
        "gymnasium>=0.29",
        "stable-baselines3>=2.1",
        "torch>=2.0",
        "pyyaml>=6.0",
        "matplotlib>=3.7",
    ],
    python_requires=">=3.10",
)
