import os
import sys
from os.path import join
from distutils.core import setup
from distutils.extension import Extension
import numpy

#Specify whether to use Cython for installation
USECYTHON = True

cythonFilesDir = join('CyLP', 'cy')
cppFilesDir = join('CyLP', 'cpp')

try:
    CoinDir = os.environ['COIN_INSTALL_DIR']
except:
    raise Exception('Please set the environment variable COIN_INSTALL_DIR ' +
                    'to the location of the COIN installation')

def get_libs():
    '''
    Return a list of distinct library names used by ``dependencies``.
    '''
    with open(join(CoinDir, 'share', 'coin',
                   'doc', 'Cbc', 'cbc_addlibs.txt')) as f:
        link_line = f.read()
        if operatingSystem == 'windows':
            libs = [flag[:-4] for flag in link_line.split() if 
                    flag.endswith('.lib')]
        else:
            libs = [flag[2:] for flag in link_line.split() if
                    flag.startswith('-l')]
    return libs

operatingSystem = sys.platform
if 'linux' in operatingSystem:
    operatingSystem = 'linux'
elif 'darwin' in operatingSystem:
    operatingSystem = 'mac'
elif 'win' in operatingSystem:
    operatingSystem = 'windows'

libs = get_libs()
print libs
libDirs = ['.', join('.', cythonFilesDir), join(CoinDir, 'lib'),
           join('.', cythonFilesDir), join(CoinDir, 'lib', 'intel')]
includeDirs = [join('.', cppFilesDir), join('.', cythonFilesDir),
                join(CoinDir, 'include', 'coin'),
                numpy.get_include(), '.']

cmdclass = {}
if USECYTHON:
    from Cython.Distutils import build_ext
    from Cython.Distutils import extension
    Extension = extension.Extension
    import Cython.Compiler.Options
    Cython.Compiler.Options.annotate = True
    cmdclass.update({'build_ext': build_ext})
    fileext = '.pyx'
else:
    fileext = '.cpp'


if operatingSystem == 'mac':
    extra_link_args = ['-Wl,-framework', '-Wl,Accelerate']
elif operatingSystem == 'linux':
    extra_link_args = ['-lrt']
else:
    extra_link_args = []

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

ext_modules += [Extension('CyLP.cy.CyClpDualRowPivotBase',
                          sources=[join(cppFilesDir,
                                        'IClpDualRowPivotBase.cpp'),
                          join(cythonFilesDir, 'CyClpDualRowPivotBase' +
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
                          join(cppFilesDir, 'IClpDualRowPivotBase.cpp'),
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
                          extra_link_args=extra_link_args), ] #,
                          #pyrex_gdb=True), ]

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

ext_modules += [Extension('CyLP.cy.CyDualPivotPythonBase',
                          sources=[join(cppFilesDir,
                                         'IClpDualRowPivotBase.cpp'),
                                   join(cythonFilesDir, 'CyDualPivotPythonBase' +
                                        fileext)],
                          language='c++',
                          include_dirs=includeDirs,
                          library_dirs=libDirs,
                          libraries=libs,
                          extra_compile_args=extra_compile_args,
                          extra_link_args=extra_link_args), ]


setup(name='CyLP',
      packages=['CyLP.cy', 'CyLP.py'],
      cmdclass=cmdclass,
      ext_modules=ext_modules)
