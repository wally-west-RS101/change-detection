import numpy as np

import glob,os
import rasterio


input_path_out_cf = '/home/skymap/data/Bahrain_change/allqc/label_bong_nha_lech/Test_delete/kq cu'
input_path_hight_building='/home/skymap/data/Bahrain_change/allqc/label_bong_nha_lech/Test_delete/t1819hight'
out_dir = '/home/skymap/data/Bahrain_change/allqc/label_bong_nha_lech/Test_delete/out_delete_bong'
for fp_img_out_change in glob.glob(os.path.join(input_path_out_cf,'*.tif')):
    full_name = os.path.basename(fp_img_out_change)
    name, _ = os.path.splitext(full_name)

    path_out_hight_building = os.path.join(input_path_hight_building,f'{name}.tif')

    with rasterio.open(fp_img_out_change, 'r') as src1:
        meta = src1.meta 
        img_out_cf = src1.read()


    with rasterio.open(path_out_hight_building,'r') as src2:
        img_out_hight_building = src2.read()
    img_out = np.zeros_like(img_out_cf)

    img_out_hight_building[img_out_hight_building == 1] = 3

    img_out = img_out_cf + img_out_hight_building
    img_out[img_out !=1 ] = 0

    out_raster = os.path.join(out_dir, f'{name}.tif')
    with rasterio.open(out_raster, 'w',compress='lzw',**meta) as dst:
        dst.write(img_out)