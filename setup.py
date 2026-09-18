from setuptools import setup

setup(
    name='ama.py',
    author='David N White',
    description='ama with jax and optax',
    version='0.0.1',
    url='https://github.com/portalgun/AMA.py.git',
    install_requires=[
        'optax @ https://github.com/google-deepmind/optax.git@b8c2e133319480509576a7280f851e9d6ec7dccf',
        'Filter @ https://github.com/portalgun/Filter.py.git',
        'numpy>=2.0.2',
        'jax>=0.4.35',
        'scikit-learn>=1.5',
        'matplotlib>=3.9.2',
        'scipy>=1.14.1',
        'PyYAML>=6.0'
    ]
)
