from setuptools import setup,find_packages
import glob
if __name__ == "__main__":
    setup(
        package_dir={'': 'src'},
        packages=find_packages('src'),
        scripts=glob.glob('scripts/*.py'),)