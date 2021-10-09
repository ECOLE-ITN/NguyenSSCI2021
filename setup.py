from setuptools import setup

setup(
    name='BO4ML',
    version='0.1.2',
    packages=['BanditOpt', 'Component','Component.mHyperopt'],
    url='http://hyperparameter.ml',
    license='MIT',
    author='Duc Anh Nguyen',
    author_email='d.a.nguyen@liacs.leidenuniv.nl',
    description='Divide and conquer strategy for Full model selection (Combined Model selection and Hyperparameter optimisation)',
    install_requires=['pandas', 'numpy', 'scipy', 'scikit-learn', 'joblib', 'dill','hyperopt']
)
