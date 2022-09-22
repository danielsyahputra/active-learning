import os
import gdown

def dir_exists(dir_name: str) -> bool:
    return os.path.isdir(dir_name)

def change_dir_name(before_name: str, after_name: str) -> None: 
    if dir_exists(before_name):
        os.rename(before_name, after_name)

def download_data(dir_name: str = "data") -> None:
    if not dir_exists(dir_name):
        os.mkdir(dir_name)
    os.chdir(dir_name)
    gdown.download(
        "https://drive.google.com/uc?id=1mjpzO99PcNKf_mR3G62ZQiYb274WfME0", quiet=False
    )
    os.system("tar -xf data.tar.gz")
    os.remove("data.tar.gz")
    try:
        change_dir_name(before_name="amazon", after_name="domain1")
        change_dir_name(before_name="dslr", after_name="domain2")
        change_dir_name(before_name="webcam", after_name="domain3")
    except FileNotFoundError:
        print("Can't find the directory!")
    os.chdir("..")