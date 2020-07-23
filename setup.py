from setuptools import setup

setup(
    name='BO4ML',
    version='0.1.0',
    packages=['BanditOpt', 'Component', 'Component.BayesOpt'],
    url='http://hyperparamter.ml',
    license='MIT',
    author='Duc Anh Nguyen',
    author_email='d.a.nguyen[at]liacs.leidenuniv[dot]nl',
    description='Machine learning - Hyperparameter optimisation tool (MIP-EGO4ML upgrade)',
    install_requires=['pandas', 'numpy', 'scipy', 'scikit-learn', 'joblib', 'dill','BayesOpt']
)
