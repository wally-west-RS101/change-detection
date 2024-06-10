import geopandas as gp
import os, glob

input_dir_shp_A = '/home/skymap/data/Bahrain_change/out_chang_fmer_test_s2223/baocao_5set/ALL_SHP_RESULT/multi'
input_dir_shp_B = '/home/skymap/data/Bahrain_change/out_chang_fmer_test_s2223/baocao_5set/ALL_SHP_RESULT/nomulti'
# out = '/home/skymap/data/Bahrain_change/out_chang_fmer_test_s2223/baocao_5set/ALL_SHP_RESULT/tmp/out_intersec'
out_final = '/home/skymap/data/Bahrain_change/out_chang_fmer_test_s2223/baocao_5set/ALL_SHP_RESULT/tmp/final'
aoi_water = '/home/skymap/data/Bahrain_change/out_chang_fmer_test_s2223/baocao_5set/ALL_SHP_RESULT/shp_water/AOI_WATER.shp'
def remove_water(shp_result, AOI_watwer,out_shp):
    shpfile_multiple_polygons = shp_result
    # shpfile_multiple_polygons = gp.read_file(shp_result)
    shpfile_single_polygon = gp.read_file(AOI_watwer)
    single_polygon_geometry = shpfile_single_polygon.geometry.iloc[0]
    selected_polygons = shpfile_multiple_polygons[shpfile_multiple_polygons.geometry.intersects(single_polygon_geometry)]   
    selected_polygons.to_file(out_shp, driver='ESRI Shapefile')

    return out_shp
# for fp_img_a in glob.glob(os.path.join(input_dir_shp_A,'*.shp')):
#     full_name = os.path.basename(fp_img_a)
#     print(fp_img_a)
#     name_b = name.replace('_multi','')
#     fp_shp_b = os.path.join(input_dir_shp_B,f'{name_b}.shp')
#     print(fp_shp_b)
#     gdf_A = gp.read_file(fp_img_a)
#     gdf_B = gp.read_file(fp_shp_b)
#     intersection_result = gp.overlay(gdf_A, gdf_B, how='intersection')
#     out_fn = os.path.join(out_final,f"{name_b}.shp")
#     remove_water(intersection_result,aoi_water,out_fn)


if __name__ == "__main__" :
    # path_shp_out = '/home/skymap/data/Bahrain_change/allqc/BAOCAO_27_2/KQ_shadow/ALL_SHP/SET_1819.shp'

    # shp = gp.read_file(path_shp_out)
    # path_water = '/home/skymap/data/Bahrain_change/out_chang_fmer_test_s2223/baocao_5set/ALL_SHP_RESULT/shp_water/AOI_WATER.shp'
    # remove_water(shp,path_water, path_shp_out)

    input_shp_dir = '/home/skymap/data/Bahrain_change/allqc/a/kq/sd_new'
    path_water = '/home/skymap/data/Bahrain_change/out_chang_fmer_test_s2223/baocao_5set/ALL_SHP_RESULT/shp_water/AOI_WATER.shp'
    for path_shp_out in glob.glob(os.path.join(input_shp_dir,'*.shp')):
        print('dang xu li ',path_shp_out)
        shp = gp.read_file(path_shp_out)
        
        remove_water(shp,path_water, path_shp_out)