import subprocess
import pkg_resources

# Read requirements.txt and install packages
with open('requirements.txt', 'r') as f:
    packages = f.read().splitlines()

for package in packages:
    subprocess.check_call(["python", '-m', 'pip', 'install', package])

# Check if packages are installed
installed_packages = pkg_resources.working_set
installed_packages_list = sorted(["%s==%s" % (i.key, i.version)
   for i in installed_packages])

for package in packages:
    if package in installed_packages_list:
        print(f'{package} has been installed successfully.')
    else:
        print(f'{package} was not installed successfully.')