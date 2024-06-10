from predict_big import predict_cf,predict_shadow
from processing import *
import rasterio
from config import Config
import numpy as np
import geopandas as gp

if __name__ == "__main__":

   
    path_dir_before_image =  '/home/skymap/big_data/Giang_workspace/Gitlab/change_detection_storage/test_data/set23'
    path_dir_after_image = '/home/skymap/big_data/Giang_workspace/Gitlab/change_detection_storage/test_data/set24'
    dir_out = '/home/skymap/big_data/Giang_workspace/Gitlab/change_detection_storage/test_outputs/set2324_newVIT_weight'
    dir_tmp = os.path.join(dir_out,'tmp')
    dir_out_shp = os.path.join(dir_out,'out_shp')
    dir_out_shp_adam = os.path.join(dir_out,'out_shp_adam')
    dir_out_all = os.path.join(dir_out,'out_all')
    out_path_cf_adam = ''
    out_path_cf = ''
    out_path_shadow = ''
    out_path_water = ''
    out_path_cd_sgb = ''
    cfg = Config() 
    if not os.path.exists(dir_out):
        os.mkdir(dir_out)
    if not os.path.exists(dir_tmp):
        os.mkdir(dir_tmp)
    if not os.path.exists(dir_out_shp):
        os.mkdir(dir_out_shp)
    if not os.path.exists(dir_out_shp_adam):
        os.mkdir(dir_out_shp_adam)
    if not os.path.exists(dir_out_all):
        os.mkdir(dir_out_all)
   
    path_da_predict = dir_out_shp
 
    list_path_dapredict = []
    for fp_path_da_pre in glob.glob(os.path.join(path_da_predict,'*.shp')):
        f_name = os.path.basename(fp_path_da_pre)
        name , _ = os.path.splitext(f_name)
        list_path_dapredict.append(name)
    print('da_pred',len(list_path_dapredict))
    for fp_bf in glob.glob(os.path.join(path_dir_before_image,"*.tif")):
        full_name = os.path.basename(fp_bf)
        name_bf , _ = os.path.splitext(full_name)
        name_af = name_bf.replace('23','24')
    
        if name_af in list_path_dapredict:
            print('da_predict',name_af)
            continue
        print('dang xu li',name_af)
      
        fp_af = os.path.join(path_dir_after_image,f'{name_af}.tif')
       
        out_path_cf = os.path.join(dir_tmp,f'{name_af}_cf.tif')
        out_path_cf_adam = os.path.join(dir_tmp,f'{name_af}_cf_adam.tif')
        out_path_shadow = os.path.join(dir_tmp,f'{name_af}_shadow.tif')
        
        out_path_water = os.path.join(dir_tmp,f'{name_af}_water.tif')
        out_path_cd_sgd = os.path.join(dir_tmp,f'{name_af}_cd_sgd.tif')
        out_path_cd_adam = os.path.join(dir_tmp,f'{name_af}_cd_adam.tif')
        #predict_cd
        print('dang predict_cf_change_detection')
        # out_path_cf = predict_cf(fp_bf, fp_af,cfg, out_path_cf,cfg.check_point_path_sgd,flag='sgd')
        out_path_cf_adam = predict_cf(fp_bf, fp_af,cfg,out_path_cf_adam,flag='adam')
        print('dang predict_shadow')
        out_path_shadow = predict_shadow(fp_bf, fp_af, out_path_shadow,cfg,cfg.path_model_shadow)
        print('dang predict_water')
        out_path_water = predict_shadow(fp_bf, fp_af, out_path_water,cfg,cfg.path_model_water)
        #mophology , rasterto vecto
        print('dang xu li raster')
        # out_path_cd_sgd = processing_raster(out_path_cf,out_path_shadow,out_path_water,out_path_cd_sgd,cfg )
        out_path_cd_adam = processing_raster(out_path_cf_adam,out_path_shadow,out_path_water,out_path_cd_adam,cfg )

        # out_path_cd_sgd_shp = os.path.join(dir_out_shp,f'{name_af}.shp')
        out_path_cd_adam_shp = os.path.join(dir_out_shp_adam,f'{name_af}.shp')
        # out_path_cd_sgd_shp = raster_to_vecto(out_path_cd_sgd, out_path_cd_sgd_shp, cfg.min_area_shp)
        out_path_cd_adam_shp = raster_to_vecto(out_path_cd_adam,out_path_cd_adam_shp,cfg.min_area_shp)
        min_area_holes = 5000
        print('dang xu li shp')
        
        
        if out_path_cd_adam_shp is not None :
            # out_path_shp_remove_holes = remove_holes(out_path_cd_sgd_shp, out_path_cd_sgd_shp, cfg.min_area_holes)
            out_path_shp_remove_holes_adam = remove_holes(out_path_cd_adam_shp,out_path_cd_adam_shp, cfg.min_area_holes)

            out_path_shp_all = os.path.join(dir_out_all,f'{name_af}.shp')
            # out_path_shp_all = intersec_shp(out_path_cd_sgd_shp,out_path_cd_adam_shp,out_path_shp_all)
            if out_path_shp_remove_holes_adam == 0:
                print(f'ko co su thay doi_{name_af}')
        else:
            print('ko co su thay doi')
            continue
