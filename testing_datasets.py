import datasets
import os

dataset_path = os.path.join(os.path.dirname(__file__), 'ILSVRC/')
label_type = 'train'
data = datasets.DLCVDataset(dataset_path,label_type)

print("sample 0: ", data.__getitem__(0))
#print("sample 1: ", data.__getitem__(1))
#print("sample 34: ", data.__getitem__(34))
print("sample 4999: ", data.__getitem__(4999))
print("sample 5000: ", data.__getitem__(5000))
print("sample 5001: ", data.__getitem__(5001))
print("")