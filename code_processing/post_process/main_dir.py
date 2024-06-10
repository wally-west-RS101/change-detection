from mophor import Morphology, delete_cloud_shadow
from raster_to_vecto import remove_holes, raster_to_vecto
from shp_processing import select_polygon_from_intersec, merge_shapefile
import os,glob
import geopandas as gp
from tqdm import *
import rasterio
import numpy as np
import shutil
if __name__ == "__main__" :

    input_dir_change = '/home/skymap/data/Bahrain_change/allqc/KETQUAMOI_EPOC_250_305/KQ205/'
    sub_dirs = [f.path for f in os.scandir(input_dir_change) if f.is_dir()]
    input_shadow = '/home/skymap/data/Bahrain_change/allqc/KETQUAMOI_EPOC_250_305/KQshadow'

    out_dir_ket_qua_full = '/home/skymap/data/Bahrain_change/allqc/KETQUAMOI_EPOC_250_305/out_305/out_205'
    for dir_change in sub_dirs:
        
        # print(dir_change)
        name_dir = os.path.basename(dir_change)
        
        dir_shadow = os.path.join(input_shadow,name_dir)
        print('dang xu li 2 thu muc')
        print(dir_change)
        print(dir_shadow)
        # print(dir_shadow)
    

        input_dir_raster = dir_change
        input_dir_shadow = dir_shadow
        input_dir_tmp = os.path.join(out_dir_ket_qua_full,name_dir +'/tmp')
        out_dir_shp = os.path.join(out_dir_ket_qua_full,name_dir +'/out_shp')
        
        if not os.path.exists(input_dir_tmp):
                os.makedirs(input_dir_tmp, exist_ok=True)
        if not os.path.exists(out_dir_shp):
                os.makedirs(out_dir_shp, exist_ok=True)
        kernel_size = 11
        for fp_img in  glob.glob(os.path.join(input_dir_raster,'*.tif')):
            full_name = os.path.basename(fp_img)
            name, _ = os.path.splitext(full_name)

            tmp_dir_raster_mophology = os.path.join(input_dir_tmp,'mophology_raster')
            if not os.path.exists(tmp_dir_raster_mophology):
                os.makedirs(tmp_dir_raster_mophology, exist_ok=True)
            
            out_raster_mophology = os.path.join(tmp_dir_raster_mophology,f'{name}.tif')

            #shadow
            input_shadow_img_path = os.path.join(input_dir_shadow, f'{name}.tif')
            with rasterio.open(input_shadow_img_path,'r') as shd:
                img_shd = shd.read()
                img_shd = img_shd.transpose(1,2,0)
                img_shd[img_shd==1] = 3
            with rasterio.open(fp_img, 'r') as src:
                img_out_cf = src.read()
                meta = src.meta
                # print(img.shape)
                
                img_out_cf = img_out_cf.transpose(1,2,0)
                image_out = np.zeros_like(img_out_cf)
                image_out = img_shd + img_out_cf
                
                image_out = delete_cloud_shadow(image_out)
                image_out = Morphology(image_out, kernel_size)
                # print(img.shape)
                
                with rasterio.open(out_raster_mophology,'w',compress='lzw',**meta) as dst:
                    dst.write(image_out[np.newaxis,:,:])
        
        list_shp_file = []
        
        for fp_raster_mopho in glob.glob(os.path.join(tmp_dir_raster_mophology,'*.tif')):
            
            tmp_dir_shapefile = os.path.join(input_dir_tmp,'shapefile')
            if not os.path.exists(tmp_dir_shapefile):
                os.makedirs(tmp_dir_shapefile, exist_ok=True)
            
            full_name = os.path.basename(fp_raster_mopho)
            name, _ = os.path.splitext(full_name)
            out_path = os.path.join(tmp_dir_shapefile,f'{name}.shp')
            # print(out_path)
            min_area = 40
            
            shp_f = raster_to_vecto(fp_raster_mopho, out_path, min_area)
            
            data = gp.read_file(shp_f)
        
            if len(data) > 1:
                list_shp_file.append(data)
        
        out_dir_merge_shp = os.path.join(out_dir_shp,'merge')
        if not os.path.exists(out_dir_merge_shp):
                os.makedirs(out_dir_merge_shp, exist_ok=True)
        out_path_merge_shp = merge_shapefile(list_shp_file, out_dir_merge_shp, 'merge.shp')

        min_area_holes = 5000
        out_path_shp_remove_holes = remove_holes(out_path_merge_shp, out_path_merge_shp, min_area_holes)
        # shutil.rmtree(input_dir_tmp)

