def get_config(dataset_name):
    if dataset_name == 'Indian':
        image_size = 7
        near_band = 7
        window_size = 25

    elif dataset_name == 'Pavia':
        image_size = 7
        near_band = 7
        window_size = 13
    elif dataset_name == 'Houston':
        image_size = 3
        near_band = 5
        window_size = 18
    elif dataset_name == 'Salinas':
        image_size = 7
        near_band = 5
        window_size = 25

    else:
        raise ValueError("Unkknow dataset")



    return image_size,near_band,window_size
