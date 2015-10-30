'''
This module contains code that copies CBC libraries into CyLP's source,
and make CyLP binaries' refernces to them relative.
'''
from __future__ import print_function
import distutils.util
import sys
import os
from os.path import join, basename, exists
from os import makedirs
import subprocess


coin_dir = os.environ['COIN_INSTALL_DIR']
cbc_libs_dir = join(coin_dir, 'lib')

platform_dir = 'lib.%s-%s.%s' % (distutils.util.get_platform(),
                                    sys.version_info.major,
                                    sys.version_info.minor)

relative_path = join('@loader_path', '..', 'cbclibs', platform_dir)

cylp_libs_dir = join('build', platform_dir, 'cylp', 'cy')

def install_name_tool(binaryFile, lib_origin_path, lib_new_path):
        os.system('install_name_tool -change  %s %s %s' % (lib_origin_path, lib_new_path, binaryFile))

def fixBinary(file):
        s = subprocess.check_output('otool -L %s' % file, shell=True)
        lib_origin_paths = [libline.decode('utf-8').split()[0] for libline in s.splitlines()[1:] if 'Cbc' in libline.decode('utf-8')]
        for lib_origin_path in lib_origin_paths:
                lib = basename(lib_origin_path)
                install_name_tool(file, lib_origin_path, join(relative_path, lib))

def copy_in_cbc_libs():
        print('Copying CBC libs into CyLP...')
        dest_dir = join('cylp', 'cbclibs', platform_dir)
        if not exists(dest_dir):
                print('Creating',  dest_dir)
                makedirs(dest_dir)
        os.system('cp %s %s' % (join(cbc_libs_dir, '*.dylib'), dest_dir))

def fixAll():
    copy_in_cbc_libs()
    for file in os.listdir(cylp_libs_dir):
        if file.endswith('so'):
                fullpath = join(cylp_libs_dir, file)
                print('Fixing %s...' % fullpath)
                fixBinary(fullpath)



