from setuptools import setup

setup(
    name='ama.py',
    author='David N White',
    description='ama with jax and optax',
    version='0.0.1',
    url='https://github.com/portalgun/AMA.py.git',
    packages=['ama'],
    py_modules=['Filter'],
    install_requires=[
        'optax>=0.2.8',                  # tested with 0.2.8 (optax.projections l2_sphere/l2_ball)
        'numpy>=2.0.2',
        'jax>=0.4.35',
        'scikit-learn>=1.5',
        'matplotlib>=3.9.2',
        'scipy>=1.14.1',
        'PyYAML>=6.0'
    ]
)
