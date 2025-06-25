import os
import sys
import platform
from os.path import join, abspath, dirname
import numpy
from subprocess import check_output

from setuptools import setup
from setuptools import Extension

VERSION = open(join('cylp', 'VERSION')).read().strip()

cythonFilesDir = join('cylp', 'cy')
cppFilesDir = join('cylp', 'cpp')

operatingSystem = sys.platform
if 'linux' in operatingSystem:
    operatingSystem = 'linux'
elif 'darwin' in operatingSystem:
    operatingSystem = 'mac'
    mac_ver = platform.mac_ver()[0]
elif 'win' in operatingSystem:
    operatingSystem = 'windows'


CoinDir = None

libs = []
libDirs = []
incDirs = []

try:
    if len(sys.argv) > 1 and (sys.argv[1] == "sdist" or sys.argv[1] == "egg_info"):
        # Do not need CoinDir
        pass
    else:
        CoinDir = os.environ['COIN_INSTALL_DIR']
except:
    # If user didn't supply location, then try pkg-config
    try:
        for p in ['cbc','cgl','osi-clp','clp','osi','coinutils']:
            flags = (check_output(['pkg-config', '--libs', p])
                     .strip().decode('utf-8'))
            for flag in flags.split():
                if flag.startswith('-l') and flag[2:] not in libs:
                    libs.append(flag[2:]) 
                if flag.startswith('-L') and flag[2:] not in libDirs:
                    libDirs.append(flag[2:]) 
            flags = (check_output(['pkg-config', '--cflags', p])
                     .strip().decode('utf-8'))
            for flag in flags.split():
                if flag.startswith('-I') and flag[2:] not in incDirs:
                    incDirs.append(flag[2:]) 
    except:
        # If pkg-config fails, then look for an installed Cbc
        try:
            location = dirname(
                check_output(['which', 'cbc']).strip()).decode('utf-8')
            CoinDir = abspath(join(location, ".."))
        except:
            #Otherwise, raise an exception
            raise Exception('''
            Could not find location of COIN installation.
            Please ensure that either 
            * COIN_INSTALL_DIR is set to the location of the installation,
            * PKG_CONFIG_PATH points to the location of the .pc files, or
            * The cbc executable is in your executable path and is installed
            at the same location as the libraries. 
            ''')

if CoinDir != None:
    # We come here if user supplied the installation directory or pkg-config failed
    if operatingSystem == 'windows':
        if os.path.exists(join(CoinDir, 'lib', 'Cbc.lib')):
            libs = ['CbcSolver', 'Cbc', 'Cgl', 'OsiClp',
                    'Clp', 'Osi', 'CoinUtils']
        else:
            libs = ['libCbcSolver', 'libCbc', 'libCgl', 'libOsiClp',
                    'libClp', 'libOsi', 'libCoinUtils']
    else:
        libs = ['CbcSolver', 'Cbc', 'Cgl', 'OsiClp', 'Clp', 'Osi',
                'CoinUtils']
        
    libDirs = [join(CoinDir, 'lib')]
    incDirs = [join(CoinDir, 'include', 'coin')] 
        
#Take care of Ubuntu case
if 'CbcSolver' not in libs:
    if operatingSystem == 'windows':
        libs.append('libCbcSolver')
    else:
        libs.append('CbcSolver')

libDirs.extend(['.', join('.', cythonFilesDir)])
try:
    libDirs.append(join(CoinDir, 'lib'))
except:
    pass

if operatingSystem == 'windows':
    try:
        libDirs.append(join(CoinDir, 'lib', 'intel'))
    except:
        pass
    
incDirs.extend([join('.', cppFilesDir), join('.', cythonFilesDir),
                numpy.get_include(), '.'])
try:
    incDirs.extend([join(CoinDir, 'include', 'coin')])
except:
    pass

cmdclass = {}

from Cython.Distutils import build_ext
from Cython.Distutils import extension
Extension = extension.Extension
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True
cmdclass.update({'build_ext': build_ext})

extra_compile_args = ['-w', '-std=c++11']
ext_modules = []

if operatingSystem == 'mac':
    extra_link_args = ['-Wl,-framework', '-Wl,Accelerate', '-headerpad_max_install_names']
elif operatingSystem == 'linux':
    extra_link_args = ['-lrt']
else:
    # Assuming Windows
    extra_link_args = []
    extra_compile_args += ['/EHsc']

ext_modules += [Extension('cylp.cy.CyClpPrimalColumnPivotBase',
                          sources=[
                              join(cppFilesDir, 'IClpPrimalColumnPivotBase.cpp'),
                              join(cythonFilesDir, 'CyClpPrimalColumnPivotBase.pyx'),
                          ],
                          language='c++',
                          include_dirs=incDirs,
                          library_dirs=libDirs,
                          libraries=libs,
                          extra_compile_args=extra_compile_args,
                          extra_link_args=extra_link_args), ]

ext_modules += [Extension('cylp.cy.CyClpDualRowPivotBase',
                          sources=[
                              join(cppFilesDir, 'IClpDualRowPivotBase.cpp'),
                              join(cythonFilesDir, 'CyClpDualRowPivotBase.pyx'),
                          ],
                          language='c++',
                          include_dirs=incDirs,
                          library_dirs=libDirs,
                          libraries=libs,
                          extra_compile_args=extra_compile_args,
                          extra_link_args=extra_link_args), ]

ext_modules += [Extension('cylp.cy.CyCglCutGeneratorBase',
                          sources=[
                              join(cppFilesDir, 'ICglCutGeneratorBase.cpp'),
                              join(cythonFilesDir, 'CyCglCutGeneratorBase.pyx'),
                          ],
                          language='c++',
                          include_dirs=incDirs,
                          library_dirs=libDirs,
                          libraries=libs,
                          extra_compile_args=extra_compile_args,
                          extra_link_args=extra_link_args), ]

ext_modules += [Extension('cylp.cy.CyOsiCuts',
                          sources=[
                              join(cppFilesDir, 'IOsiCuts.cpp'),
                              join(cythonFilesDir, 'CyOsiCuts.pyx'),
                          ],
                          language='c++',
                          include_dirs=incDirs,
                          library_dirs=libDirs,
                          libraries=libs,
                          extra_compile_args=extra_compile_args,
                          extra_link_args=extra_link_args), ]

ext_modules += [Extension('cylp.cy.CyOsiSolverInterface',
                          sources=[
                              join(cythonFilesDir, 'CyOsiSolverInterface.pyx'),
                          ],
                          language='c++',
                          include_dirs=incDirs,
                          library_dirs=libDirs,
                          libraries=libs,
                          extra_compile_args=extra_compile_args,
                          extra_link_args=extra_link_args), ]

ext_modules += [Extension('cylp.cy.CyCglTreeInfo',
                          sources=[
                              join(cythonFilesDir, 'CyCglTreeInfo.pyx'),
                          ],
                          language='c++',
                          include_dirs=incDirs,
                          library_dirs=libDirs,
                          libraries=libs,
                          extra_compile_args=extra_compile_args,
                          extra_link_args=extra_link_args), ]

ext_modules += [Extension('cylp.cy.CyCoinIndexedVector',
                          sources=[
                              join(cppFilesDir, 'ICoinIndexedVector.cpp'),
                              join(cythonFilesDir, 'CyCoinIndexedVector.pyx'),
                          ],
                          language='c++',
                          include_dirs=incDirs,
                          library_dirs=libDirs,
                          libraries=libs,
                          extra_compile_args=extra_compile_args,
                          extra_link_args=extra_link_args), ]

ext_modules += [Extension('cylp.cy.CyCoinPackedMatrix',
                          sources=[
                              join(cppFilesDir, 'ICoinPackedMatrix.cpp'),
                              join(cythonFilesDir, 'CyCoinPackedMatrix.pyx'),
                          ],
                          language='c++',
                          include_dirs=incDirs,
                          library_dirs=libDirs,
                          libraries=libs,
                          extra_compile_args=extra_compile_args,
                          extra_link_args=extra_link_args), ]

ext_modules += [Extension('cylp.cy.CyCoinModel',
                          sources=[
                              join(cythonFilesDir, 'CyCoinModel.pyx'),
                          ],
                          language='c++',
                          include_dirs=incDirs,
                          library_dirs=libDirs,
                          libraries=libs,
                          extra_compile_args=extra_compile_args,
                          extra_link_args=extra_link_args), ]

ext_modules += [Extension('cylp.cy.CyCoinMpsIO',
                          sources=[
                              join(cppFilesDir, 'ICoinMpsIO.cpp'),
                              join(cythonFilesDir, 'CyCoinMpsIO.pyx'),
                          ],
                          language='c++',
                          include_dirs=incDirs,
                          library_dirs=libDirs,
                          libraries=libs,
                          extra_compile_args=extra_compile_args,
                          extra_link_args=extra_link_args), ]

ext_modules += [Extension('cylp.cy.CyCgl',
                          sources=[
                              join(cythonFilesDir, 'CyCgl.pyx'),
                          ],
                          language='c++',
                          include_dirs=incDirs,
                          library_dirs=libDirs,
                          libraries=libs,
                          extra_compile_args=extra_compile_args,
                          extra_link_args=extra_link_args), ]

ext_modules += [Extension('cylp.cy.CyCbcNode',
                          sources=[
                              join(cppFilesDir, 'ICbcNode.cpp'),
                              join(cythonFilesDir, 'CyCbcNode.pyx'),
                          ],
                          language='c++',
                          include_dirs=incDirs,
                          library_dirs=libDirs,
                          libraries=libs,
                          extra_compile_args=extra_compile_args,
                          extra_link_args=extra_link_args), ]

ext_modules += [Extension('cylp.cy.CyCbcModel',
                          sources=[
                              join(cppFilesDir, 'ICbcModel.cpp'),
                              join(cppFilesDir, 'CbcCompareUser.cpp'),
                              join(cppFilesDir, 'ICbcNode.cpp'),
                              join(cythonFilesDir, 'CyCbcModel.pyx'),
                          ],
                          language='c++',
                          include_dirs=incDirs,
                          library_dirs=libDirs,
                          libraries=libs,
                          extra_compile_args=extra_compile_args,
                          extra_link_args=extra_link_args), ]

ext_modules += [Extension('cylp.cy.CyClpSimplex',
                          sources=[
                              join(cppFilesDir, 'IClpPrimalColumnPivotBase.cpp'),
                              join(cppFilesDir, 'IClpDualRowPivotBase.cpp'),
                              join(cppFilesDir, 'IClpSimplex.cpp'),
                              #join(cppFilesDir, 'ICbc.cpp'),
                              join(cppFilesDir, 'ICbcModel.cpp'),
                              join(cppFilesDir, 'CbcCompareUser.cpp'),
                              join(cppFilesDir, 'IClpSimplexPrimal_Wolfe.cpp'),
                              join(cppFilesDir, 'IClpSimplexPrimal.cpp'),
                              join(cppFilesDir, 'IClpPackedMatrix.cpp'),
                              join(cythonFilesDir, 'CyClpSimplex.pyx'),
                          ],
                          language='c++',
                          include_dirs=incDirs,
                          library_dirs=libDirs,
                          libraries=libs,
                          extra_compile_args=extra_compile_args,
                          extra_link_args=extra_link_args), ] #,
                          #pyrex_gdb=True), ]

ext_modules += [Extension('cylp.cy.CyPEPivot',
                           sources=[
                               join(cppFilesDir, 'IClpPrimalColumnPivotBase.cpp'),
                               join(cythonFilesDir, 'CyPEPivot.pyx'),
                          ],
                          language='c++',
                          include_dirs=incDirs,
                          library_dirs=libDirs,
                          libraries=libs,
                          extra_compile_args=extra_compile_args,
                          extra_link_args=extra_link_args), ]

ext_modules += [Extension('cylp.cy.CyWolfePivot',
                           sources=[
                               join(cppFilesDir, 'IClpPrimalColumnPivotBase.cpp'),
                               join(cythonFilesDir, 'CyWolfePivot.pyx'),
                          ],
                          language='c++',
                          include_dirs=incDirs,
                          library_dirs=libDirs,
                          libraries=libs,
                          extra_compile_args=extra_compile_args,
                          extra_link_args=extra_link_args), ]

ext_modules += [Extension('cylp.cy.CyDantzigPivot',
                          sources=[
                              join(cppFilesDir, 'IClpPrimalColumnPivotBase.cpp'),
                              join(cythonFilesDir, 'CyDantzigPivot.pyx'),
                          ],
                          language='c++',
                          include_dirs=incDirs,
                          library_dirs=libDirs,
                          libraries=libs,
                          extra_compile_args=extra_compile_args,
                          extra_link_args=extra_link_args), ]

ext_modules += [Extension('cylp.cy.CyTest',
                          sources=[
                              join(cythonFilesDir, 'CyTest.pyx'),
                          ],
                          language='c++',
                          include_dirs=incDirs,
                          library_dirs=libDirs,
                          libraries=libs,
                          extra_compile_args=extra_compile_args,
                          extra_link_args=extra_link_args), ]

ext_modules += [Extension('cylp.cy.CyPivotPythonBase',
                          sources=[
                              join(cppFilesDir, 'IClpPrimalColumnPivotBase.cpp'),
                              join(cythonFilesDir, 'CyPivotPythonBase.pyx'),
                          ],
                          language='c++',
                          include_dirs=incDirs,
                          library_dirs=libDirs,
                          libraries=libs,
                          extra_compile_args=extra_compile_args,
                          extra_link_args=extra_link_args), ]

ext_modules += [Extension('cylp.cy.CyDualPivotPythonBase',
                          sources=[
                              join(cppFilesDir, 'IClpDualRowPivotBase.cpp'),
                              join(cythonFilesDir, 'CyDualPivotPythonBase.pyx'),
                          ],
                          language='c++',
                          include_dirs=incDirs,
                          library_dirs=libDirs,
                          libraries=libs,
                          extra_compile_args=extra_compile_args,
                          extra_link_args=extra_link_args), ]

ext_modules += [Extension('cylp.cy.CyCutGeneratorPythonBase',
                          sources=[
                              join(cppFilesDir, 'ICglCutGeneratorBase.cpp'),
                              join(cppFilesDir, 'IOsiCuts.cpp'),
                              join(cythonFilesDir, 'CyCutGeneratorPythonBase.pyx'),
                          ],
                          language='c++',
                          include_dirs=incDirs,
                          library_dirs=libDirs,
                          libraries=libs,
                          extra_compile_args=extra_compile_args,
                          extra_link_args=extra_link_args), ]

# Add language level directive per https://stackoverflow.com/a/58116368
for e in ext_modules:
    e.cython_directives = {'language_level': "3"} #all are Python-3

extra_files = ['cpp/*.hpp', 'cpp/*.h', 'cy/*.pxd', 'VERSION']

setup(packages=['cylp', 'cylp.cy', 'cylp.py', 'cylp.py.pivots',
                'cylp.py.modeling', 'cylp.py.utils', 'cylp.py.mip',
                'cylp.py.QP'],
      cmdclass={'build_ext': build_ext},
      ext_modules=ext_modules,
      zip_safe = False,
      package_data={"cylp": extra_files})
