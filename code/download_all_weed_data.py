import wget
import tarfile
import argparse
import glob
import os
import sys
from os import path

def progressBar(current, total, width=80):
  progress_message = "File progress: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
  sys.stdout.write("\r" + progress_message)
  sys.stdout.flush()

def main(weed_file, root_folder, base_url, remote_path):
    root_folder = root_folder + '/fielddata'
    os.makedirs(root_folder + remote_path, exist_ok = True)
    base_url = base_url + remote_path

    download_list = None
    with open(weed_file) as downloads_file:
            download_list = downloads_file.readlines()

    for file in download_list:
        file_fixed = file.split('\n')[0]
        print("")
        if (path.exists(root_folder + remote_path + file_fixed)):
            print('Skipping ' + base_url + '/' + file_fixed)
        else:
            print('Downloading ' + base_url + '/' + file_fixed)
            wget.download(base_url + '/' + file_fixed, out=root_folder + remote_path, bar=progressBar)

    #extract each folder
    tar_files = glob.glob(root_folder + remote_path + '/**/*.tar.gz', recursive=True)
    for file in tar_files:
        f = tarfile.open(name=file, mode='r:gz')
        print("")
        print('Extracting ' + file )
        f.extractall(path=root_folder + '/')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='download and extaction script for weed data')
    parser.add_argument('-f', '--file', type=str, help='file with list of weed data in tar.gz format', required=True)
    parser.add_argument('-r', '--root_download_folder', type=str, help='download root folder', required=True)
    parser.add_argument('-b', '--base_url', type=str, default='https://openweeds.linkoping-ri.se/data', help='base url for tar files', required=False)
    parser.add_argument('-p', '--remote_path', type=str, default='/tractor-33-zipped', help='download artefacts instead of images', required=False)

    args = parser.parse_args()
    
    try:
        fh = open('/code/LICENCE-DATA.md', 'r')
    except:
        print("Could not find licence file. Exiting.")
        exit(1)
        
    file_contents = fh.read()
    print(file_contents)
    fh.close()
    
    print("")
    print("-----------------------------------------------------------------------------------------------")
    print("")
    
    try:
        answer = os.environ['ACCEPT_ARTEFACTS_LICENSE']
    except:
        answer = "NO"       
    
    if (answer == "YESPLEASE"):
        print("License agreed upon. Will download artefact.")
        print("")
        print("-----------------------------------------------------------------------------------------------")
        print("")
        main(weed_file=args.file, root_folder=args.root_download_folder, base_url=args.base_url, remote_path=args.remote_path)
    else:
        print("License not agreed upon. Will NOT download artefact.")
        print("If you agree to the above license, please set ACCEPT_ARTEFACTS_LICENSE=YESPLEASE in 'env.list'")
        print("Exit.")