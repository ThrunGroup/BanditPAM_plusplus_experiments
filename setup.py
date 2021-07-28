'''
Based on https://github.com/pybind/pybind11_benchmark/blob/master/setup.py
'''

import sys
import sysconfig
import os
import tempfile
import setuptools
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

__version__ = '0.0.35'

class get_pybind_include(object):
    '''
    Helper class to determine the pybind11 include path.
    The purpose of this class is to postpone importing pybind11
    until it is actually installed via setup's setup_requires arg,
    so that the ``get_include()`` method can be invoked.
    '''

    def __str__(self):
        import pybind11
        return pybind11.get_include()

class get_numpy_include(object):
    '''
    Helper class to determine the numpy include path
    The purpose of this class is to postpone importing numpy
    until it is actually installed via setup's setup_requires arg,
    so that the ``get_include()`` method can be invoked.
    '''

    def __str__(self):
        import numpy
        return numpy.get_include()


def has_flag(compiler, flagname):
    '''
    Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    '''
    with tempfile.NamedTemporaryFile('w', suffix='.cpp', delete=False) as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        fname = f.name
    try:
        compiler.compile([fname], extra_postargs=[flagname])
    except setuptools.distutils.errors.CompileError:
        return False
    finally:
        try:
            os.remove(fname)
        except OSError:
            print("Warning: Received an OSError")
    return True


def cpp_flag(compiler):
    '''
    Return the -std=c++[11/14/17] compiler flag.
    The newer version is prefered over c++11 (when it is available).
    '''

    if sys.platform == 'darwin':
        # Assuming that on OSX, building with clang
        flags = ['-std=c++17', '-std=c++14', '-std=c++11']
    else:
        # Assuming that on linux, building on a manylinux image (old) with gcc
        flags = ['-std=c++1y']
    
    for flag in flags:
        if has_flag(compiler, flag):
            return flag

    raise RuntimeError('Unsupported compiler -- at least C++11 support '
                       'is needed!')


def check_brew_package(pkg_name):
    brew_cmd = ['brew', '--prefix', pkg_name]
    process = subprocess.Popen(brew_cmd, stdout=subprocess.PIPE)
    output, _error = process.communicate()
    if output.decode() == '':
        raise Exception('Error: Need to install %s via homebrew! Please run `brew install %s`' % (pkg_name, pkg_name))
    return output.decode().strip()


def check_brew_installation():
    brew_cmd = ['which', 'brew']
    process = subprocess.Popen(brew_cmd, stdout=subprocess.PIPE)
    output, _error = process.communicate()
    if output.decode() == '':
        raise Exception('Error: Need to install homebrew! Please see https://brew.sh')


def check_numpy_installation():
    try:
        import numpy
    except ModuleNotFoundError:
        raise Exception('Need to install numpy!')


def install_check_mac():
    # Make sure homebrew is installed
    check_brew_installation()

    # Make sure numpy is installed
    check_numpy_installation()

    # Check that LLVM clang, libomp, and armadillo are installed
    llvm_loc = check_brew_package('llvm') # We need to use LLVM clang since Apple's clang doesn't support OpenMP
    _libomp_loc = check_brew_package('libomp')
    _arma_loc = check_brew_package('armadillo')
    
    # Set compiler to LLVM clang on Mac, since Apple clang doesn't support OpenMP
    os.environ["CC"] = os.path.join(llvm_loc, 'bin', 'clang')


def check_omp_install_linux():
    # TODO: Need to get exact compiler name and version to check this
    # Check compiler version is gcc>=6.0.0 or clang>=X.X.X
    pass


def check_armadillo_install_linux():
    # Since armadillo is a C++ extension, just check if it exists
    cmd = ['find', '/', '-iname', 'armadillo']
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    output, _error = process.communicate()
    if output.decode() == '':
        print("Warning: Armadillo may not be installed. \
            Please build it from", os.path.join('BanditPAM', 'headers', 'carma', 'third_party', 'armadillo-code'))
    return output.decode().strip()


def check_linux_package_installation(pkg_name):
    cmd = ['dpkg', '-s', pkg_name]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    output, _error = process.communicate()
    if output.decode() == '':
        raise Exception('Error: Need to install %s via homebrew! \
            Please ensure all dependencies are installed via your package manager (apt, yum, etc.): \
            build-essential checkinstall libreadline-gplv2-dev libncursesw5-dev libssl-dev \
            libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev' % (pkg_name))
    return output.decode().strip()


def install_check_linux():
    # Make sure linux packages are installed
    dependencies = [
        'build-essential',
        'checkinstall',
        'libreadline-gplv2-dev',
        'libncursesw5-dev',
        'libssl-dev',
        'libsqlite3-dev',
        'tk-dev',
        'libgdbm-dev',
        'libc6-dev',
        'libbz2-dev',
        'libffi-dev',
        'zlib1g-dev',
        ]

    for dep in dependencies:
        check_linux_package_installation(dep)

    # Make sure numpy is installed
    check_numpy_installation()

    # Check openMP is installed
    check_omp_install_linux()

    # Check armadillo is installed
    check_armadillo_install_linux()

class BuildExt(build_ext):
    '''
    A custom build extension for adding compiler-specific options.
    '''
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }
    l_opts = {
        'msvc': [],
        'unix': [],
    }
    
    if sys.platform == 'darwin':
        install_check_mac()

        darwin_opts = ['-stdlib=libc++', '-mmacosx-version-min=10.7', '-O3']
        c_opts['unix'] += darwin_opts
        l_opts['unix'] += darwin_opts
    elif sys.platform == 'linux' or sys.platform =='linux2':
        install_check_linux()

        linux_opts = ['-O3']
        c_opts['unix'] += linux_opts
        l_opts['unix'] += linux_opts


    def build_extensions(self):
        ct = self.compiler.compiler_type
        
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])
        
        opts.append('-Wno-register')
        opts.append(cpp_flag(self.compiler))
        opts.append('-O3') # Add it here as well, in case of Windows installation
        opts.append('-fopenmp')
        
        #TODO: Change OMP library library name based on gcc vs clang instead of based on OS
        if sys.platform == 'darwin':
            # We assume that if the user is on OSX, then they are building with clang (required above)
            link_opts.append('-lomp')
        else:
            # We assume that if the user is on linux, then they are building with gcc
            link_opts.append('-lgomp')
        
        if ct == 'unix':
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        
        for ext in self.extensions:
            ext.define_macros = [('VERSION_INFO', '"{}"'.format(self.distribution.get_version()))]
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts
        
        build_ext.build_extensions(self)


#TODO: Change OMP library library name based on gcc vs clang instead of based on OS
if sys.platform == 'linux' or sys.platform == 'linux2':
    include_dirs=[
            get_pybind_include(),
            get_numpy_include(),
            'headers',
            '/usr/local/include',
            '/usr/local/include/carma',
            '/usr/local/include/carma/carma',
        ]
    # We assume that if the user is on linux, then they are building with gcc
    libraries=['armadillo', 'gomp']
else: # OSX
    include_dirs=[
            get_pybind_include(),
            get_numpy_include(),
            'headers',
            'headers/carma/include',
            'headers/carma/include/carma',
            'headers/carma/include/carma/carma',
            '/usr/local/include',
        ]
    # We assume that if the user is on OSX, then they are building with clang (required above)
    libraries = ['armadillo', 'omp']

ext_modules = [
    Extension(
        'BanditPAM',
        [os.path.join('src', 'kmeds_pywrapper.cpp'), os.path.join('src', 'kmedoids_ucb.cpp')],
        include_dirs=include_dirs,
        library_dirs=[
            '/usr/local/lib',
        ],
        libraries=libraries,
        language='c++1y', #TODO: modify this based on cpp_flag(compiler)
        extra_compile_args=['-static-libstdc++'],
    ),
]

with open(os.path.join('docs', 'long_desc.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='BanditPAM',
    version=__version__,
    author='Mo Tiwari and James Mayclin',
    maintainer="Mo Tiwari",
    author_email='motiwari@stanford.edu',
    url='https://github.com/ThrunGroup/BanditPAM',
    description='BanditPAM: A state-of-the-art, high-performance k-medoids algorithm.',
    long_description=long_description,
    ext_modules=ext_modules,
    setup_requires=[
        'pybind11>=2.5.0',
        'numpy>=1.18',
    ],
    data_files=[('docs', [os.path.join('docs', 'long_desc.rst')])],
    include_package_data=True,
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    headers=[os.path.join('headers', 'kmedoids_ucb.hpp')],
)
