import requests
import os
import zipfile
import subprocess
import fnmatch
import stat
import re
import xmltodict
from colorama import Fore, Back, Style, init


class Spark(object):
    _spark_versions = {}
    SPARK_REPO_URL = 'https://api.github.com/repos/apache/spark'
    HOME_DIR = os.path.expanduser('~')
    SVM_DIR = '.svm'
    SVM_ACTIVE_SPARK_DIR = 'active_spark'
    init()

    @classmethod
    def spark_versions(cls):
        if not cls._spark_versions:
            cls._spark_versions = {v['name']: v['zipball_url'] for v in
                                   requests.get(Spark.SPARK_REPO_URL + '/tags').json()}
        return sorted(cls._spark_versions)

    @staticmethod
    def print_version_list():
        installed_versions = Spark.list_installed_versions()
        active_version = Spark.active_version()
        print("\nApache Spark Available Versions\n")
        for version in Spark.spark_versions():
            if version[1:] in installed_versions:
                if version[1:] == active_version:
                    print(Fore.GREEN + "   [*] {}".format(version[1:]))
                else:
                    print(Fore.GREEN + "   [ ] {}".format(version[1:]))
            else:
                print("   [ ] {}".format(version[1:]))
            print(Style.RESET_ALL)

    @staticmethod
    def list_installed_versions():
        versions = []
        for f in os.listdir(Spark.svm_path()):
            regex = re.match('^v([\d.]{5})$', f)
            if regex:
                versions.append(regex.groups()[0])

        return sorted(versions)

    @staticmethod
    def print_installed_version_list():
        versions = Spark.list_installed_versions()
        active_version = Spark.active_version()
        print("\nApache Spark Versions (installed at {})\n".format(Spark.svm_path()))
        for version in versions:
            if version == active_version:
                print(Fore.GREEN + "   [*] {}".format(version))
            else:
                print(Fore.GREEN + "   [ ] {}".format(version))
        print(Style.RESET_ALL + "\n   * Active")

    @staticmethod
    def active_version():
        with open(os.path.join(Spark.svm_spark_path(), 'pom.xml')) as f:
            pom_file = xmltodict.parse(f.read())
            return pom_file['project']['version']

    @staticmethod
    def download(url, localfile):
        r = requests.get(url, stream=True)
        os.makedirs(os.path.dirname(localfile), exist_ok=True)
        with open(localfile, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    @staticmethod
    def has_java():
        return b'Runtime Environment' in subprocess.check_output(["java", "-version"], stderr=subprocess.STDOUT)

    @staticmethod
    def check_source_install_dependencies():
        if not Spark.has_java():
            raise SparkInstallationError('Spark requires Java. Please install before continuing.')
        return True

    @staticmethod
    def install(version):
        Spark.check_source_install_dependencies()
        Spark.download_source(version)
        Spark.unzip(Spark.svm_version_path(version) + '.zip')
        Spark.rename_unzipped_folder(version)
        Spark.build_from_source(version)

    @staticmethod
    def activate_spark(version, **kwargs):
        Spark.check_svm_spark_dir_exists()
        path = '/usr/local/bin'
        spark_version_bin_path = os.path.join(Spark.svm_version_path(version), 'bin')
        relink_attempted = kwargs.get('relink_attempted', False)

        try:
            for f in os.listdir(spark_version_bin_path):
                spark_cmd_file = os.path.join(spark_version_bin_path, f)

                # if not fnmatch.fnmatch(f, '*.*'):
                os.symlink(spark_cmd_file, os.path.join(path, f))

                Spark.chmod_add_excute(spark_cmd_file)
        except FileExistsError:
            if relink_attempted:
                raise FileExistsError

            Spark.deactivate_spark(version)
            Spark.activate_spark(version, relink_attempted=True)

        spark_version_path = Spark.svm_version_path(version)
        active_spark_path = Spark.svm_spark_path()
        try:
            for f in os.listdir(spark_version_path):
                spark_cmd_file = os.path.join(spark_version_path, f)
                os.symlink(spark_cmd_file, os.path.join(active_spark_path, f))
        except FileExistsError:
            if relink_attempted:
                raise FileExistsError

            Spark.deactivate_spark(version)
            Spark.activate_spark(version, relink_attempted=True)

    @staticmethod
    def deactivate_spark(version):
        path = '/usr/local/bin'
        spark_version_bin_path = os.path.join(Spark.svm_version_path(version), 'bin')
        for f in os.listdir(spark_version_bin_path):
            try:
                os.unlink(os.path.join(path, f))
            except FileNotFoundError:
                pass

        spark_version_path = Spark.svm_version_path(version)
        active_spark_path = Spark.svm_spark_path()
        for f in os.listdir(spark_version_path):
            try:
                os.unlink(os.path.join(active_spark_path, f))
            except FileNotFoundError:
                pass

    @staticmethod
    def svm_spark_path():
        return os.path.join(Spark.svm_path(), Spark.SVM_ACTIVE_SPARK_DIR)

    @staticmethod
    def check_svm_spark_dir_exists():
        svm_spark_dir = Spark.svm_spark_path()
        if not os.path.exists(svm_spark_dir):
            os.mkdir(svm_spark_dir)

    @staticmethod
    def build_from_source(version, **kwargs):
        """
        Builds specified Spark version from source.
        :param version:
        :param kwargs:
        :return: (Integer) Status code of build/mvn command.
        """
        mvn = os.path.join(Spark.svm_version_path(version), 'build', 'mvn')
        Spark.chmod_add_excute(mvn)
        p = subprocess.Popen([mvn, '-DskipTests', 'clean', 'package'], cwd=Spark.svm_version_path(version))
        p.wait()
        return p.returncode

    @staticmethod
    def chmod_add_excute(filename):
        """
        Adds execute permission to file.
        :param filename:
        :return:
        """
        st = os.stat(filename)
        os.chmod(filename, st.st_mode | stat.S_IEXEC)

    @staticmethod
    def download_source(version):
        """
        Download Spark version. Uses same name as release tag without the leading 'v'.
        :param version: Version number to download.
        :return: None
        """
        local_filename = 'v{}.zip'.format(Spark.svm_version_path(version))
        Spark.download(Spark.spark_versions()['v{}'.format(version)], local_filename)

    @staticmethod
    def svm_version_path(version):
        """
        Path to specified spark version. Accepts semantic version numbering.
        :param version: Spark version as String
        :return: String.
        """
        return os.path.join(Spark.HOME_DIR, Spark.SVM_DIR, 'v{}'.format(version))

    @staticmethod
    def svm_path():
        """
        Path to local Spark verions folder. Defaults to user_home/.svm
        :return: String
        """
        return os.path.join(Spark.HOME_DIR, Spark.SVM_DIR)

    @staticmethod
    def unzip(filename):
        """
        Unzips specified file into ~/.svm
        :param filename:
        :return:
        """
        with zipfile.ZipFile(filename, "r") as zip_ref:
            zip_ref.extractall(Spark.svm_path())
        return True

    @staticmethod
    def rename_unzipped_folder(version):
        """
        Renames unzipped spark version folder to the release tag.
        :param version: version from release tag.
        :return:
        """

        for filename in os.listdir(Spark.svm_path()):
            if fnmatch.fnmatch(filename, 'apache-spark-*'):
                return os.rename(os.path.join(Spark.svm_path(), filename), Spark.svm_version_path(version))

        raise SparkInstallationError("Unable to find unzipped Spark folder in {}".format(Spark.svm_path()))


class SparkInstallationError(AssertionError):
    def __init__(self, *args, **kwargs):
        AssertionError.__init__(self, *args, **kwargs)
