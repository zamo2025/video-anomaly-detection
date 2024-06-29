import tarfile

path = './data/UCSD_Anomaly_Dataset.tar.gz'
destination = './data/'

with tarfile.open(path, 'r:gz') as tar:
    tar.extractall(path=destination)