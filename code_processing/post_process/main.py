from mophor import Morphology, delete_cloud_shadow
from raster_to_vecto import remove_holes, raster_to_vecto
from shp_processing import select_polygon_from_intersec, merge_shapefile
import os,glob
import geopandas as gp
from tqdm import *
import rasterio
import numpy as np
import shutil
from intersecshp import remove_water
if __name__ == "__main__" :
    
    input_dir = '/home/skymap/data/Bahrain_change/allqc/BAOCAO_quytrinh_21-5/1718_sgd'
    sub_dirs = [f.path for f in os.scandir(input_dir) if f.is_dir()]
    

    for dir_change in sub_dirs:
        print('dang xu li',dir_change)
        input_dir_raster = dir_change
        
        input_dir_tmp = os.path.join(dir_change,'tmp')
        out_dir_shp = os.path.join(dir_change,'out_shp')
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

            with rasterio.open(fp_img, 'r') as src:
                img = src.read()
                meta = src.meta
                # print(img.shape)
                
                img = img.transpose(1,2,0)
                # img[img==255] = 1
                img = delete_cloud_shadow(img)
                img = Morphology(img, kernel_size)
                # print(img.shape)
                
                with rasterio.open(out_raster_mophology,'w',compress='lzw',**meta) as dst:
                    dst.write(img[np.newaxis,:,:])
        
        list_shp_file = []
        
        for fp_raster_mopho in glob.glob(os.path.join(tmp_dir_raster_mophology,'*.tif')):
            print(fp_raster_mopho)
            tmp_dir_shapefile = os.path.join(input_dir_tmp,'shapefile')
            if not os.path.exists(tmp_dir_shapefile):
                os.makedirs(tmp_dir_shapefile, exist_ok=True)
            
            full_name = os.path.basename(fp_raster_mopho)
            name, _ = os.path.splitext(full_name)
            out_path = os.path.join(tmp_dir_shapefile,f'{name}.shp')
            # print(out_path)
            min_area = 70
            
            shp_f = raster_to_vecto(fp_raster_mopho, out_path, min_area)
            if shp_f is not None:
                 
                data = gp.read_file(shp_f)
            else:
                 continue
        
            if len(data) > 1:
                list_shp_file.append(data)
        
        out_dir_merge_shp = os.path.join(out_dir_shp,'merge')
        if not os.path.exists(out_dir_merge_shp):
                os.makedirs(out_dir_merge_shp, exist_ok=True)
        out_path_merge_shp = merge_shapefile(list_shp_file, out_dir_merge_shp, 'merge.shp')

        min_area_holes = 5000
        out_path_shp_remove_holes = remove_holes(out_path_merge_shp, out_path_merge_shp, min_area_holes)
        # shp = gp.read_file(out_path_shp_remove_holes)
        # remove_water(shp,path_shp_water,out_path_shp_remove_holes)
        print('doneeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee')
        # shutil.rmtree(input_dir_tmp)

