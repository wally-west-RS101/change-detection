class Config:
    n_class = 4,
    check_point_path_sgd = '/home/skymap/big_data/Giang_workspace/Gitlab/change_detection_storage/weight_changedetection/changedetection/check_point_model_sgd/best_ckpt.pt'
    check_point_path_adam = '/home/skymap/big_data/Giang_workspace/Gitlab/change_detection_storage/weight_changedetection/changedetection/checkpoint_model_adam/best_ckpt_adam_606.pt'
    model_path_classify = '/home/skymap/big_data/Giang_workspace/Gitlab/change_detection_storage/weight_changedetection/classify/classify_459.pth'
    img_size = 256
    num_bands = 4
    path_model_shadow ='/home/skymap/big_data/Giang_workspace/Gitlab/change_detection_storage/weight_changedetection/detectshadow_water/model_shadow.h5'
    path_model_water = '/home/skymap/big_data/Giang_workspace/Gitlab/change_detection_storage/weight_changedetection/detectshadow_water/model_water.h5'
    embeddim = 64
    kenel_size_mopho = 11
    min_area_shp = 70
    min_area_holes = 5000