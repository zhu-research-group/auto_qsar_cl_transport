import os


def directory_check(data_dir):

    if not os.path.exists(os.path.join(data_dir, 'predictions')):
        os.makedirs(os.path.join(data_dir, 'predictions'))

    if not os.path.exists(os.path.join(data_dir, 'results')):
        os.makedirs(os.path.join(data_dir, 'results'))

    if not os.path.exists(os.path.join(data_dir, 'models')):
        os.makedirs(os.path.join(data_dir, 'models'))
