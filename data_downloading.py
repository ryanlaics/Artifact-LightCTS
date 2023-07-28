import os
import gdown
import shutil

# Dataset information
datasets = datasets = {
    'multi_step': {
        'PEMS04': {
            'data_type': 'Traffic Flow',
            'files': [
                {'name': 'PEMS04.csv', 'url': 'https://drive.google.com/uc?id=1tBarIeC_IpGf9G2quC2W1gdmM1kbk--i'},
                {'name': 'PEMS04.npz', 'url': 'https://drive.google.com/uc?id=1uLhIkbJHpqOcsL2AE5_21X9wNoHYjh8J'}
            ]
        },
        'PEMS08': {
            'data_type': 'Traffic Flow',
            'files': [
                {'name': 'PEMS08.csv', 'url': 'https://drive.google.com/uc?id=11DoIOUqmvY1dEU5XBIdWLtSghNgtG6Pf'},
                {'name': 'PEMS08.npz', 'url': 'https://drive.google.com/uc?id=1eObvPx6UQ7mRoDjJaFOboPVCx41zHPQ3'}
            ]
        },
        'METR-LA': {
            'data_type': 'Traffic Speed',
            'files': [
                {'name': 'adj_mx_metr_la.pkl', 'url': 'https://drive.google.com/uc?id=1zpdZzHGIO8BbUHiMsz8MP3xfBDgWDNLn'},
                {'name': 'train.npz', 'url': 'https://drive.google.com/uc?id=1-c_y0py-EKTXXZbX6lrHKpiOwqEdd0zn'},
                {'name': 'test.npz', 'url': 'https://drive.google.com/uc?id=1-dpZnnCcRZiWRbTGbcs4TauqhRGGnzRz'},
                {'name': 'val.npz', 'url': 'https://drive.google.com/uc?id=1-dGrZEQUfowadQEKizCXvcokdeisoShC'}
            ]
        },
        'PEMS-BAY': {
            'data_type': 'Traffic Speed',
            'files': [
                {'name': 'adj_mx_pems_bay.pkl', 'url': 'https://drive.google.com/uc?id=1ECjE8EXUfRuGI8xXMtZxyBZuUqNm8rbh'},
                {'name': 'train.npz', 'url': 'https://drive.google.com/uc?id=1-fNqehwpqmj7tfUMBLAILVs-G_DPHxGO'},
                {'name': 'test.npz', 'url': 'https://drive.google.com/uc?id=1-m4Dh1DMHbLPK6GWUNjGyNKPiYK2uLzl'},
                {'name': 'val.npz', 'url': 'https://drive.google.com/uc?id=1-iu_CBVZv41ceFMVk-EXKNN2Cyi0OKCm'}
            ]
        }
    },
    'single_step': {
        'Solar': {
            'data_type': 'Solar Power Production',
            'name': 'solar.txt',
            'url': 'https://drive.google.com/uc?id=1TP6xDPXmf923YdhdRPdD1VwqtLATQqkS'
        },
        'Electricity': {
            'data_type': 'Electricity Consumption',
            'name': 'electricity.txt',
            'url': 'https://drive.google.com/uc?id=1x9nBW-RAXrubHCWeG6JLMtUaXl2iQznX'
        }
    }
}

# Ensure the data directory exists
if not os.path.exists('data'):
    os.mkdir('data')

# Download the datasets
for step_type in datasets.keys():
    print(f'Downloading {step_type} datasets...')
    for dataset_name, dataset_info in datasets[step_type].items():
        if step_type == 'multi_step':
            # Ensure the dataset directory exists
            if not os.path.exists(f'data/{dataset_name}'):
                os.mkdir(f'data/{dataset_name}')
            for file in dataset_info['files']:
                print(f'Downloading {file["name"]} for dataset {dataset_name}...')
                gdown.download(file['url'], output=f"{file['name']}", quiet=False)
                print(f'{file["name"]} for dataset {dataset_name} has been downloaded.')
                # Move the file to the correct directory
                shutil.move(f"{file['name']}", f"data/{dataset_name}/{file['name']}")
        else:
            print(f'Downloading', dataset_info['name'])
            gdown.download(dataset_info['url'], output=f"{dataset_info['name']}", quiet=False)
            print(dataset_info['name'], 'has been downloaded.')
            # Move the file to the data directory
            shutil.move(f"{dataset_info['name']}", f"data/{dataset_info['name']}")

print('All datasets have been downloaded and moved to the data directory.')
