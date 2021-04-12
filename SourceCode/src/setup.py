# Authors: Paul Boniol, Themis Palpanas
# Date: 08/07/2020
# copyright retained by the authors
# algorithms protected by patent application FR2005261
# code provided as is, and can be used only for research purposes
#
# Reference using:
#
# P. Boniol and T. Palpanas, Series2Graph: Graph-based Subsequence Anomaly Detection in Time Series, PVLDB (2020)
#
# P. Boniol and T. Palpanas and M. Meftah and E. Remy, GraphAn: Graph-based Subsequence Anomaly Detection, demo PVLDB (2020)
#


from setuptools import setup, Extension


setup(name='series2graph',
      version='0.1',
      description='series2graph alpha', #TODO: create a description
      author='Paul Boniol',
      license='Univerity of Paris and EDF R&D',
      packages=['series2graph'],
      install_requires=[
          'numpy',
          'matplotlib',
          'plotly',
          'scipy',
          'pandas',
          'tqdm',
          'networkx',
          'sklearn',	
      ],
      zip_safe=False)
