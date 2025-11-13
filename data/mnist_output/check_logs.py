from pathlib import Path


dataset_list = ['mnist', 'cifar10', 'sinewave']



for dataset_name in dataset_list:
    output_path : Path = Path.cwd() / f'{dataset_name}_output'

    if not output_path.exists():
        continue

    # 获取当前目录下的所有文件夹
    folders = [item.name for item in output_path.iterdir() if item.is_dir()]

    for folder in folders:
        target_path : Path = output_path / f'{folder}' / 'logs'
        if target_path.exists():
            print(f'{dataset_name}->{folder} 可能有问题')
            # # 读取内容，分析原因
            # file_path = target_path / 'generation.log'
            # with open(file_path, 'r', encoding='utf-8') as file:
            #     content = file.read()
            #     if content.find('LocallyConnected2D') != -1:
            #         continue
            #     if content.find('LocallyConnected1D') != -1:
            #         continue
            #     print(f'{dataset_name}->{folder} 可能有问题')
