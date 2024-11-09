import os
import sys
import shutil


if __name__=='__main__':

    base_path = os.getcwd()

    package_name = sys.argv[1]
    print('Creating packfluff for package : ', package_name)
    print('Located at path                : ', base_path)

    description = '...'
    email = 'nilsaxelhenningsson@gmail.com'
    name = 'Axel Henningsson'
    version = "0.0.0"
    dependencies = [
      "numpy",
      "scipy",
      "matplotlib",
      "pandas",
      "xfab",
      "dfxm"
      ]

    package_path = os.path.join(base_path, package_name)
    if not os.path.isdir(package_path): os.mkdir(package_path)

    with open(os.path.join(package_path,'__init__.py'), 'a'):
        pass

    test_path = os.path.join(base_path, 'tests')
    if not os.path.isdir(test_path): os.mkdir(test_path)

    with open(os.path.join(test_path,'__init__.py'), 'a'):
        pass

    toml = """[build-system]
    requires = ["setuptools>=42", "wheel"]
    build-backend = "setuptools.build_meta"

    [tool.setuptools]
    py-modules = ["_package_name_"]

    [project]
    name = "_package_name_"
    version = "_version_"
    description = "_description_"
    authors = [
      { name = "{_myname_}", email = "{_email_}" }
    ]
    dependencies = [_dependencies_]
    """

    toml = toml.replace("_myname_", name).replace("_email_", email)
    toml = toml.replace("_package_name_", package_name).replace("_version_", version)
    toml = toml.replace("_description_", description)

    if len(dependencies)>0:
      dd = '"'+dependencies[0]+'"'+',\n'
      for d in dependencies[1:]:
          dd += '"'+d+'"'+',\n'
      toml = toml.replace("_dependencies_", dd)
    else:
        toml = toml.replace("_dependencies_", "")

    with open(os.path.join(base_path, 'pyproject.toml'), 'w') as t:
        t.write(toml)

    print(toml)

    source_folder = os.path.abspath(os.path.join(os.path.abspath(__file__), '..'))
    destination_folder = os.path.join(base_path)
    shutil.copytree(source_folder, destination_folder, dirs_exist_ok=True)

    for fname in ['index.rst', 'conf.py']:
        print(os.path.join(base_path, 'docs', 'source', fname))
        with open(os.path.join(base_path, 'docs', 'source', fname), 'w', encoding='utf-8') as file:
            file_data = file.read()
          
        file_data = file_data.replace('_package_name_', package_name)
        
        with open(os.path.join(base_path, 'docs', 'source', fname), 'r', encoding='utf-8') as file:
            file.write(file_data)