import os
import glob
import geopandas as gp

from tqdm import tqdm
from config import Config
from predict_big import predict_cf,predict_shadow
from processing import processing_raster, raster_to_vecto, remove_holes

if __name__ == "__main__":

   
    before_images_fol = '/home/skymap/big_data/Giang_workspace/Gitlab/change_detection_storage/test_data/set23'
    after_images_fol = '/home/skymap/big_data/Giang_workspace/Gitlab/change_detection_storage/test_data/set24'
    dir_out = '/home/skymap/big_data/Giang_workspace/Gitlab/change_detection_storage/test_outputs/set2324_newVIT_weight'
    
    dir_tmp = os.path.join(dir_out,'tmp')
    dir_out_shp_adam = os.path.join(dir_out,'out_shp_adam')
    if not os.path.exists(dir_out):
        os.mkdir(dir_out)
    if not os.path.exists(dir_tmp):
        os.mkdir(dir_tmp)
    if not os.path.exists(dir_out_shp_adam):
        os.mkdir(dir_out_shp_adam)
        
    cfg = Config() 
    
    predicted_names_list = []
    
    # get all predicted images
    for predicted_path in glob.glob(os.path.join(dir_out_shp_adam,'*.shp')):
        full_name = os.path.basename(predicted_path)
        name_before , _ = os.path.splitext(full_name)
        predicted_names_list.append(name_before)
    print(f'Total {len(predicted_names_list)} images predicted.')
    
    # pbar
    pbar = tqdm(glob.glob(os.path.join(before_images_fol,"*.tif")), desc='Predicting...')
    
    for file_before in pbar:
        full_name = os.path.basename(file_before)
        name_before , _ = os.path.splitext(full_name)
        pbar.set_postfix(file=name_before)
        name_after = name_before.replace('23','24')
        # name_after = 'K3_20240521075540_64084_06181175_Mosaic_2nd Set Data_COG_uint8'
        # if name_after in predicted_names_list:
        #     print(f'{name_after} predicted. Continue..')
        #     continue
      
        path_after = os.path.join(after_images_fol,f'{name_after}.tif')
       
        out_path_cf_adam = os.path.join(dir_tmp,f'{name_after}_cf_adam.tif') # output of ChangeFormer model
        out_path_shadow = os.path.join(dir_tmp,f'{name_after}_shadow.tif')
        out_path_water = os.path.join(dir_tmp,f'{name_after}_water.tif')
        out_path_cd_adam = os.path.join(dir_tmp,f'{name_after}_cd_adam.tif')
        
        # predict changes with ChangeFormer and shadow, water with U2net
        print(f"{'#'*20} Predicting changes {'#'*20}")
        out_path_cf_adam = predict_cf(file_before, path_after,cfg,
                                      out_path_cf_adam,flag='adam')
        
        print(f"{'#'*20} Predicting shadow {'#'*20}")
        out_path_shadow = predict_shadow(file_before, path_after,
                                         out_path_shadow,cfg,cfg.path_model_shadow)
        print(f"{'#'*20} Predicting water {'#'*20}")
        out_path_water = predict_shadow(file_before, path_after,
                                        out_path_water,cfg,cfg.path_model_water)
        
        # Postprocess , raster to vector
        print(f"{'#'*20} Processing raster {'#'*20}")
        out_path_cd_adam = processing_raster(out_path_cf_adam,
                                             out_path_shadow,out_path_water,out_path_cd_adam,cfg )
        out_path_cd_adam_shp = os.path.join(dir_out_shp_adam,
                                            f'{name_after}.shp')
        out_path_cd_adam_shp = raster_to_vecto(out_path_cd_adam,
                                               out_path_cd_adam_shp,cfg.min_area_shp)
        min_area_holes = 5000
        
        print(f"{'#'*20} Processing shapefile {'#'*20}")
        if  out_path_cd_adam_shp is not None :
            out_path_shp_remove_holes_adam = remove_holes(out_path_cd_adam_shp,out_path_cd_adam_shp, cfg.min_area_holes) # this line doesn't contribute to the output
        
        adam_shp = gp.read_file(out_path_cd_adam_shp)
        if len(adam_shp) ==0:
            print(f"There's no change in {name_after}")
            continue
        

        