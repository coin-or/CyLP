import sys
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

operatingSystem = sys.platform
if 'linux' in operatingSystem:
    operatingSystem = 'linux'
elif 'darwin' in operatingSystem:
    operatingSystem = 'mac'
#WINDOWS??

from os.path import join

#Specify whether to use Cython for installation
USECYTHON = True

cmdclass = {}
if USECYTHON:
    cmdclass.update({'build_ext': build_ext})
    fileext = '.pyx'
else:
    fileext = '.cpp'

CoinDir = '/usr/local/Cellar/coinmp/1.6.0/'

cythonFilesDir = 'cy/'
cppFilesDir = 'cpp/'

libs = ['Clp', 'Cbc', 'Cgl', 'CoinMP', 'Osi',
        'CoinUtils', 'OsiCbc', 'OsiClp', 'CbcSolver', 'bz2']

libDirs = ['.', join('.', cythonFilesDir), join(CoinDir, 'lib')]

includeDirs = ['./cpp/', join('.', cythonFilesDir),
                join(CoinDir, 'include/coin/'),
                join(CoinDir, 'BuildTools/headers/'),
                join(CoinDir, 'Clp/src/'), numpy.get_include(), '.']

if operatingSystem == 'mac':
    extra_link_args = ['-Wl,-framework', '-Wl,Accelerate']
elif operatingSystem == 'linux':
    extra_link_args = ['-llapack', '-lblas']
else:
    #WINDOWS?
    extra_link_args = ['-llapack', '-lblas']

extra_compile_args = []
ext_modules = []

ext_modules += [Extension('CyLP.cy.CyClpPrimalColumnPivotBase',
                          sources=[join(cppFilesDir,
                                        'IClpPrimalColumnPivotBase.cpp'),
                          join(cythonFilesDir, 'CyClpPrimalColumnPivotBase' +
                               fileext)],
                          language='c++',
                          include_dirs=includeDirs,
                          library_dirs=libDirs,
                          libraries=libs,
                          extra_compile_args=extra_compile_args,
                          extra_link_args=extra_link_args), ]


ext_modules += [Extension('CyLP.cy.CyCoinIndexedVector',
                          sources=[join(cppFilesDir, 'ICoinIndexedVector.cpp'),
                          join(cythonFilesDir,
                               'CyCoinIndexedVector' + fileext)],
                          language='c++',
                          include_dirs=includeDirs,
                          library_dirs=libDirs,
                          libraries=libs,
                          extra_compile_args=extra_compile_args,
                          extra_link_args=extra_link_args), ]

ext_modules += [Extension('CyLP.cy.CyCoinPackedMatrix',
                          sources=[join(cppFilesDir, 'ICoinPackedMatrix.cpp'),
                          join(cythonFilesDir,
                               'CyCoinPackedMatrix' + fileext)],
                          language='c++',
                          include_dirs=includeDirs,
                          library_dirs=libDirs,
                          libraries=libs,
                          extra_compile_args=extra_compile_args,
                          extra_link_args=extra_link_args), ]

ext_modules += [Extension('CyLP.cy.CyCoinModel',
                          sources=[join(cythonFilesDir,
                                        'CyCoinModel' + fileext)],
                          language='c++',
                          include_dirs=includeDirs,
                          library_dirs=libDirs,
                          libraries=libs,
                          extra_compile_args=extra_compile_args,
                          extra_link_args=extra_link_args), ]

ext_modules += [Extension('CyLP.cy.CyCoinMpsIO',
                          sources=[join(cppFilesDir, 'ICoinMpsIO.cpp'),
                          join(cythonFilesDir, 'CyCoinMpsIO' + fileext)],
                          language='c++',
                          include_dirs=includeDirs,
                          library_dirs=libDirs,
                          libraries=libs,
                          extra_compile_args=extra_compile_args,
                          extra_link_args=extra_link_args), ]

ext_modules += [Extension('CyLP.cy.CyCgl',
                          sources=[join(cythonFilesDir,
                                   'CyCgl' + fileext)],
                          language='c++',
                          include_dirs=includeDirs,
                          library_dirs=libDirs,
                          libraries=libs,
                          extra_compile_args=extra_compile_args,
                          extra_link_args=extra_link_args), ]

ext_modules += [Extension('CyLP.cy.CyCbcNode',
                          sources=[join(cppFilesDir, 'ICbcNode.cpp'),
                          join(cythonFilesDir, 'CyCbcNode' + fileext)],
                          language='c++',
                          include_dirs=includeDirs,
                          library_dirs=libDirs,
                          libraries=libs,
                          extra_compile_args=extra_compile_args,
                          extra_link_args=extra_link_args), ]

ext_modules += [Extension('CyLP.cy.CyCbcModel',
                          sources=[join(cppFilesDir, 'ICbcModel.cpp'),
                                   join(cppFilesDir, 'CbcCompareUser.cpp'),
                                   join(cppFilesDir, 'ICbcNode.cpp'),
                                   join(cythonFilesDir,
                                        'CyCbcModel' + fileext)],
                          language='c++',
                          include_dirs=includeDirs,
                          library_dirs=libDirs,
                          libraries=libs,
                          extra_compile_args=extra_compile_args,
                          extra_link_args=extra_link_args), ]

ext_modules += [Extension('CyLP.cy.CyClpSimplex',
                          sources=[join(cppFilesDir,
                                        'IClpPrimalColumnPivotBase.cpp'),
                          join(cppFilesDir, 'IClpSimplex.cpp'),
                          #join(cppFilesDir, 'ICbc.cpp'),
                          join(cppFilesDir, 'ICbcModel.cpp'),
                          join(cppFilesDir, 'CbcCompareUser.cpp'),
                          join(cppFilesDir, 'IClpSimplexPrimal_Wolfe.cpp'),
                          join(cppFilesDir, 'IClpSimplexPrimal.cpp'),
                          join(cppFilesDir, 'IClpPackedMatrix.cpp'),
                          join(cythonFilesDir, 'CyClpSimplex' + fileext)],
                          language='c++',
                          include_dirs=includeDirs,
                          library_dirs=libDirs,
                          libraries=libs,
                          extra_compile_args=extra_compile_args,
                          extra_link_args=extra_link_args), ]

ext_modules += [Extension('CyLP.cy.CyPEPivot',
                           sources=[join(cppFilesDir, 
                                         'IClpPrimalColumnPivotBase.cpp'),
                                    join(cythonFilesDir, 'CyPEPivot' + fileext)],
                          language='c++',
                          include_dirs=includeDirs,
                          library_dirs=libDirs,
                          libraries=libs,
                          extra_compile_args=extra_compile_args,
                          extra_link_args=extra_link_args), ]

ext_modules += [Extension('CyLP.cy.CyWolfePivot',
                           sources=[join(cppFilesDir, 
                                         'IClpPrimalColumnPivotBase.cpp'),
                                    join(cythonFilesDir, 'CyWolfePivot' + fileext)],
                          language='c++',
                          include_dirs=includeDirs,
                          library_dirs=libDirs,
                          libraries=libs,
                          extra_compile_args=extra_compile_args,
                          extra_link_args=extra_link_args), ]

ext_modules += [Extension('CyLP.cy.CyDantzigPivot',
                          sources=[join(cppFilesDir, 
                                         'IClpPrimalColumnPivotBase.cpp'),
                                   join(cythonFilesDir, 'CyDantzigPivot' +
                                        fileext)],
                          language='c++',
                          include_dirs=includeDirs,
                          library_dirs=libDirs,
                          libraries=libs,
                          extra_compile_args=extra_compile_args,
                          extra_link_args=extra_link_args), ]

ext_modules += [Extension('CyLP.cy.CyTest',
                          sources=[join(cythonFilesDir, 'CyTest' + fileext)],
                          language='c++',
                          include_dirs=includeDirs,
                          library_dirs=libDirs,
                          libraries=libs,
                          extra_compile_args=extra_compile_args,
                          extra_link_args=extra_link_args), ]

ext_modules += [Extension('CyLP.cy.CyPivotPythonBase',
                          sources=[join(cppFilesDir, 
                                         'IClpPrimalColumnPivotBase.cpp'),
                                   join(cythonFilesDir, 'CyPivotPythonBase' +
                                        fileext)],
                          language='c++',
                          include_dirs=includeDirs,
                          library_dirs=libDirs,
                          libraries=libs,
                          extra_compile_args=extra_compile_args,
                          extra_link_args=extra_link_args), ]


setup(name='CyLP',
      packages=['CyLP.cy'],
      cmdclass=cmdclass,
      ext_modules=ext_modules)
