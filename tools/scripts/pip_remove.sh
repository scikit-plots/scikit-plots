
# https://stackoverflow.com/questions/68886239/cannot-uninstall-numpy-1-21-2-record-file-not-found
# https://github.com/andreafrancia/trash-cli
SITE_PACKAGES_FOLDER=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
echo $SITE_PACKAGES_FOLDER

ls $SITE_PACKAGES_FOLDER/numpy*

pip install trash-cli
trash-put $SITE_PACKAGES_FOLDER/numpy*

pip install --upgrade numpy
