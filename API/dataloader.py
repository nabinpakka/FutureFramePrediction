from .dataloader_taxibj import load_data as load_taxibj
from .dataloader_moving_mnist import load_data as load_mmnist
from .dataloader_traffic import load_data as load_traffic_video_data
from .dataloader_traffic_dynamic import load_data as load_traffic_dynamic_data

def load_data(dataname,batch_size, val_batch_size, data_root, num_workers, input_frames, output_frames, **kwargs):
    if dataname == 'taxibj':
        return load_taxibj(batch_size, val_batch_size, data_root, num_workers)
    elif dataname == 'mmnist':
        return load_mmnist(batch_size, val_batch_size, data_root, num_workers)
    elif dataname == 'traffic':
        return load_traffic_video_data(batch_size, val_batch_size, data_root, num_workers, input_frames, output_frames)
    elif dataname == 'traffic_dynamic':
        return load_traffic_dynamic_data(batch_size, val_batch_size, data_root, num_workers)