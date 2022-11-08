import wget
import tarfile
import argparse
import glob
import os

def main(weed_file, root_folder, base_url):    
    os.makedirs(root_folder + '/tractor-33-zipped', exist_ok = True)

    download_list = None    
    with open(weed_file) as downloads_file:            
            download_list = downloads_file.readlines()

    for file in download_list:
        file_fixed = file.split('\n')[0]
        wget.download(base_url + '/' + file_fixed, out=root_folder + '/tractor-33-zipped/')

    #extract each folder    
    tar_files = glob.glob(root_folder + '/tractor-33-zipped' + '/**/*.tar.gz', recursive=True)
    for file in tar_files:
        f = tarfile.open(name=file, mode='r:gz')  
        print('extracting ' + file )      
        f.extractall(path=root_folder + '/')




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='download and extaction script for weed data')
    parser.add_argument('-f', '--file', type=str, help='file with list of weed data in tar.gz format', required=True)
    parser.add_argument('-r', '--root_download_folder', type=str, help='download root folder', required=True)
    parser.add_argument('-b', '--base_url', type=str, default='https://openweeds.linkoping-ri.se/data/tractor-33-zipped', help='base url for tar files', required=False)    

    args = parser.parse_args()

    main(weed_file=args.file, root_folder=args.root_download_folder, base_url=args.base_url)