import os
import sys
import platform
from os.path import join
import numpy
import unicodedata

#A unicode function that is compatible with Python 2 and 3
u = lambda s: s if sys.version_info[0] > 2 else unicode(s, 'utf-8')


# Setuptools seems to get confused with c++ extensions
try:
    from setuptools import setup
    from setuptools import Extension
    from setuptools.command.install import install
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension
    from distutils.command.install import install

PROJECT = 'cylp'
VERSION = open(join('cylp', 'VERSION')).read()
URL = 'https://github.com/mpy/cylp'
AUTHOR_EMAIL = u('mehdi.towhidi@gerad.ca')
DESC = 'A Python interface for CLP, CBC, and CGL'


cythonFilesDir = join('cylp', 'cy')
cppFilesDir = join('cylp', 'cpp')

#Do "export CYLP_USE_CYTHON=" if you want to build cylp from scratch, using Cython
try:
    os.environ['CYLP_USE_CYTHON']
    USECYTHON = True
except:
    USECYTHON = False

try:
    CoinDir = os.environ['COIN_INSTALL_DIR']
except:
    from subprocess import check_output
    from os.path import abspath, dirname

    try:
        location = dirname(
            check_output(['which', 'clp']).strip()).decode('utf-8')
        CoinDir = abspath(join(location, ".."))
    except:
        raise Exception('Please set the environment variable COIN_INSTALL_DIR'
                        ' to the location of the COIN installation')

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

def getBdistFriendlyString(s):
    '''
    Solve the issue with restructuredText README
    "ordinal not in range error" when using bdist_mpkg or bdist_wininst
    '''
    return unicodedata.normalize('NFKD', u(s))



operatingSystem = sys.platform
if 'linux' in operatingSystem:
    operatingSystem = 'linux'
elif 'darwin' in operatingSystem:
    operatingSystem = 'mac'
    mac_ver = platform.mac_ver()[0]

elif 'win' in operatingSystem:
    operatingSystem = 'windows'

libs = get_libs()
#Take care of Ubuntu case
if 'CbcSolver' not in libs:
    if operatingSystem == 'windows':
        libs.append('libCbcSolver')
    else:
        libs.append('CbcSolver')

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


extra_compile_args = ['-w']
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

ext_modules += [Extension('cylp.cy.CyClpDualRowPivotBase',
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

ext_modules += [Extension('cylp.cy.CyCglCutGeneratorBase',
                          sources=[join(cppFilesDir,
                                        'ICglCutGeneratorBase.cpp'),
                            join(cythonFilesDir, 'CyCglCutGeneratorBase' +
                               fileext)],
                          language='c++',
                          include_dirs=includeDirs,
                          library_dirs=libDirs,
                          libraries=libs,
                          extra_compile_args=extra_compile_args,
                          extra_link_args=extra_link_args), ]

ext_modules += [Extension('cylp.cy.CyOsiCuts',
                          sources=[join(cppFilesDir, 'IOsiCuts.cpp'),
                            join(cythonFilesDir, 'CyOsiCuts' + fileext)],
                          language='c++',
                          include_dirs=includeDirs,
                          library_dirs=libDirs,
                          libraries=libs,
                          extra_compile_args=extra_compile_args,
                          extra_link_args=extra_link_args), ]

ext_modules += [Extension('cylp.cy.CyOsiSolverInterface',
                          sources=[join(cythonFilesDir, 'CyOsiSolverInterface' +
                                        fileext)],
                          language='c++',
                          include_dirs=includeDirs,
                          library_dirs=libDirs,
                          libraries=libs,
                          extra_compile_args=extra_compile_args,
                          extra_link_args=extra_link_args), ]

ext_modules += [Extension('cylp.cy.CyCglTreeInfo',
                          sources=[join(cythonFilesDir, 'CyCglTreeInfo' +
                                        fileext)],
                          language='c++',
                          include_dirs=includeDirs,
                          library_dirs=libDirs,
                          libraries=libs,
                          extra_compile_args=extra_compile_args,
                          extra_link_args=extra_link_args), ]

ext_modules += [Extension('cylp.cy.CyCoinIndexedVector',
                          sources=[join(cppFilesDir, 'ICoinIndexedVector.cpp'),
                          join(cythonFilesDir,
                               'CyCoinIndexedVector' + fileext)],
                          language='c++',
                          include_dirs=includeDirs,
                          library_dirs=libDirs,
                          libraries=libs,
                          extra_compile_args=extra_compile_args,
                          extra_link_args=extra_link_args), ]

ext_modules += [Extension('cylp.cy.CyCoinPackedMatrix',
                          sources=[join(cppFilesDir, 'ICoinPackedMatrix.cpp'),
                          join(cythonFilesDir,
                               'CyCoinPackedMatrix' + fileext)],
                          language='c++',
                          include_dirs=includeDirs,
                          library_dirs=libDirs,
                          libraries=libs,
                          extra_compile_args=extra_compile_args,
                          extra_link_args=extra_link_args), ]

ext_modules += [Extension('cylp.cy.CyCoinModel',
                          sources=[join(cythonFilesDir,
                                        'CyCoinModel' + fileext)],
                          language='c++',
                          include_dirs=includeDirs,
                          library_dirs=libDirs,
                          libraries=libs,
                          extra_compile_args=extra_compile_args,
                          extra_link_args=extra_link_args), ]

ext_modules += [Extension('cylp.cy.CyCoinMpsIO',
                          sources=[join(cppFilesDir, 'ICoinMpsIO.cpp'),
                          join(cythonFilesDir, 'CyCoinMpsIO' + fileext)],
                          language='c++',
                          include_dirs=includeDirs,
                          library_dirs=libDirs,
                          libraries=libs,
                          extra_compile_args=extra_compile_args,
                          extra_link_args=extra_link_args), ]

ext_modules += [Extension('cylp.cy.CyCgl',
                          sources=[join(cythonFilesDir,
                                   'CyCgl' + fileext)],
                          language='c++',
                          include_dirs=includeDirs,
                          library_dirs=libDirs,
                          libraries=libs,
                          extra_compile_args=extra_compile_args,
                          extra_link_args=extra_link_args), ]

ext_modules += [Extension('cylp.cy.CyCbcNode',
                          sources=[join(cppFilesDir, 'ICbcNode.cpp'),
                          join(cythonFilesDir, 'CyCbcNode' + fileext)],
                          language='c++',
                          include_dirs=includeDirs,
                          library_dirs=libDirs,
                          libraries=libs,
                          extra_compile_args=extra_compile_args,
                          extra_link_args=extra_link_args), ]

ext_modules += [Extension('cylp.cy.CyCbcModel',
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

ext_modules += [Extension('cylp.cy.CyClpSimplex',
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

ext_modules += [Extension('cylp.cy.CyPEPivot',
                           sources=[join(cppFilesDir,
                                         'IClpPrimalColumnPivotBase.cpp'),
                                    join(cythonFilesDir, 'CyPEPivot' + fileext)],
                          language='c++',
                          include_dirs=includeDirs,
                          library_dirs=libDirs,
                          libraries=libs,
                          extra_compile_args=extra_compile_args,
                          extra_link_args=extra_link_args), ]

ext_modules += [Extension('cylp.cy.CyWolfePivot',
                           sources=[join(cppFilesDir,
                                         'IClpPrimalColumnPivotBase.cpp'),
                                    join(cythonFilesDir, 'CyWolfePivot' + fileext)],
                          language='c++',
                          include_dirs=includeDirs,
                          library_dirs=libDirs,
                          libraries=libs,
                          extra_compile_args=extra_compile_args,
                          extra_link_args=extra_link_args), ]

ext_modules += [Extension('cylp.cy.CyDantzigPivot',
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

ext_modules += [Extension('cylp.cy.CyTest',
                          sources=[join(cythonFilesDir, 'CyTest' + fileext)],
                          language='c++',
                          include_dirs=includeDirs,
                          library_dirs=libDirs,
                          libraries=libs,
                          extra_compile_args=extra_compile_args,
                          extra_link_args=extra_link_args), ]

ext_modules += [Extension('cylp.cy.CyPivotPythonBase',
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

ext_modules += [Extension('cylp.cy.CyDualPivotPythonBase',
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

ext_modules += [Extension('cylp.cy.CyCutGeneratorPythonBase',
                          sources=[join(cppFilesDir,
                                        'ICglCutGeneratorBase.cpp'),
                                    join(cppFilesDir,
                                         'IOsiCuts.cpp'),
                                    join(cythonFilesDir, 'CyCutGeneratorPythonBase' +
                                        fileext)],
                          language='c++',
                          include_dirs=includeDirs,
                          library_dirs=libDirs,
                          libraries=libs,
                          extra_compile_args=extra_compile_args,
                          extra_link_args=extra_link_args), ]


s_README = getBdistFriendlyString(open('README.rst').read())
s_AUTHORS = u(open('AUTHORS').read())
s_LICENSE = u(open('LICENSE').read())


class customInstall(install):
    '''
    Take care of adding std:: to isspace in Cython-generated files.
    This is currently an issue for Mac OS Mavericks.
    '''
    def run(self):
        currentDir = os.path.dirname(os.path.realpath(__file__))
        from distutils.version import LooseVersion
        if operatingSystem == 'mac' and LooseVersion(mac_ver) >= LooseVersion('10.9'):
            # If std::isspace is not already replaced
            if os.system('grep -rI "std::isspace" cylp/cy/*.cpp'):
                os.system('''find %s -name "*.cpp" -print | xargs sed -i "" 's/isspace/std::isspace/g' ''' % currentDir)

        if operatingSystem == 'mac':
            from fixBinaries import platform_dir
            extra_files.append(join('cbclibs', platform_dir, '*.dylib'))

        install.run(self)

        if operatingSystem == 'mac':
            from fixBinaries import fixAll
            fixAll()


cmdclass['install'] = customInstall

extra_files = ['cpp/*.cpp', 'cpp/*.hpp', 'cpp/*.h', 'VERSION']

setup(name='cylp',
      version=VERSION,
      description=DESC,
      long_description=s_README,
      author=s_AUTHORS,
      author_email=AUTHOR_EMAIL,
      url=URL,
      license=s_LICENSE,
      packages=['cylp', 'cylp.cy', 'cylp.py', 'cylp.py.pivots', 'cylp.py.modeling',
                'cylp.py.utils', 'cylp.py.mip','cylp.py.QP'],
      cmdclass=cmdclass,
      ext_modules=ext_modules,
      install_requires=['numpy >= 1.5.0', 'scipy >= 0.10.0'],
      zip_safe = False,
      package_data={"cylp": extra_files})
