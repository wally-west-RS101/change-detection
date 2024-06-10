import geopandas as gpd
import os, glob
import pandas as pd
from tqdm import *
from shapely.geometry import Polygon



def select_polygon_from_intersec(fp_path_select_polygon, fp_path_intersec, out_path):

    shp_a = gpd.read_file(fp_path_select_polygon)
    shp_b = gpd.read_file(fp_path_intersec)
    full_name = os.path.basename(shp_a)
    name , _ = os.path.splitext(full_name)


    selected_polygons = []

    for index, polygon_a in shp_a.iterrows():
    
        intersects_with_b = any(polygon_a['geometry'].intersects(polygon_b) for _, polygon_b in shp_b.iterrows())
        

        if intersects_with_b:
            selected_polygons.append(polygon_a)

    result = gpd.GeoDataFrame(selected_polygons, crs=shp_a.crs)

    out = os.path.join(out_path,f'{name}.shp')
    result.to_file(out)
    return out


def merge_shapefile(list_shp,out_dir,name):
    data = gpd.GeoDataFrame(pd.concat(list_shp,ignore_index = True),crs = list_shp[0].crs)
    out = os.path.join(out_dir,name)
    data.to_file(out)
    return out


# def shp_processing(shp_a, shp_b,out):
#     # intersec
#     intersec = gpd.overlay(shp_a, shp_b, how='intersection')

#     #select polygon A
#     selected_from_shp_a = shp_a[shp_a.geometry.intersects(intersec.unary_union)]
#     #select_polygon B
#     selected_from_shp_b = shp_b[shp_b.geometry.intersects(intersec.unary_union)]
#     #union select
#     union_result = gpd.overlay(selected_from_shp_a, selected_from_shp_b, how='union')
#     #dislove 
#     dissolved_result = union_result.dissolve()
#     #multipart to single part
#     singlepart_gdf = dissolved_result.explode()
#     singlepart_gdf.to_file(out)

#     return out

# import geopandas as gpd
# from shapely.geometry import Polygon


# shp_a = gpd.read_file('/home/skymap/data/Bahrain_change/allqc/label_bong_nha_lech/Test_delete/kq_set1819_lan1_xoabong/SET_1819.shp')
# shp_b = gpd.read_file('/home/skymap/data/Bahrain_change/allqc/label_bong_nha_lech/Test_delete/t1819hight/out_shp/merge/merge.shp')


# for index, row in shp_a.iterrows():
    
#     polygon_a = row['geometry']
    
    
#     intersects = shp_b['geometry'].intersects(polygon_a).any()
    
    
#     if intersects:
#         shp_a.drop(index, inplace=True)


# shp_a.to_file('/home/skymap/data/Bahrain_change/allqc/label_bong_nha_lech/Test_delete/kq_set1819_lan1_xoabong/out_xoa/SET_1819.shp')

# import geopandas as gpd

# # Đọc dữ liệu từ file shapefile
# shp_a = gpd.read_file('/home/skymap/data/Bahrain_change/allqc/label_bong_nha_lech/test_code_shp/A.shp')
# shp_b = gpd.read_file('/home/skymap/data/Bahrain_change/allqc/label_bong_nha_lech/test_code_shp/B.shp')

# result = gpd.overlay(shp_a, shp_b, how='intersection')


# selected_from_shp_a = shp_a[shp_a.geometry.intersects(result.unary_union)]
# selected_from_shp_b = shp_b[shp_b.geometry.intersects(result.unary_union)]
# union_result = gpd.overlay(selected_from_shp_a, selected_from_shp_b, how='union')

# print(union_result['geometry'])
# dissolved_result = union_result.dissolve()



# singlepart_gdf = dissolved_result.explode()
# # largest_boundary = union_result.unary_union.convex_hull
# # result = gpd.GeoDataFrame(geometry=[largest_boundary], crs=union_result.crs)
# out = '/home/skymap/data/Bahrain_change/allqc/label_bong_nha_lech/test_code_shp/out5.shp'
# # In ra kết quả hoặc lưu ra file shapefile nếu cần
# singlepart_gdf.to_file(out)


# def shp_processing(shp_a, shp_b,out):
#     # intersec
#     intersec = gpd.overlay(shp_a, shp_b, how='intersection  ')

#     #select polygon A
#     selected_from_shp_a = shp_a[shp_a.geometry.intersects(intersec.unary_union)]
#     #select_polygon B
#     selected_from_shp_b = shp_b[shp_b.geometry.intersects(intersec.unary_union)]
#     #union select
#     union_result = gpd.overlay(selected_from_shp_a, selected_from_shp_b, how='union')
#     #dislove 
#     dissolved_result = union_result.dissolve()
#     #multipart to single part
#     singlepart_gdf = dissolved_result.explode()
#     singlepart_gdf.to_file(out)

#     return out


def shp_process_new(shp_a, shp_b,out):
    print('intersec')
    intersec = gpd.overlay(shp_a, shp_b, how='intersection')

    #select polygon A
   
    #select_polygon B
    print('select')

    #union select


    sindex_b = shp_b.sindex

# Tìm các đa giác của A có ít nhất một phần giao với B sử dụng Spatial Index
    result_A = []

    for idx_a, geometry_a in shp_a.iterrows():
        possible_matches_index = list(sindex_b.intersection(geometry_a['geometry'].bounds))
        possible_matches = shp_b.iloc[possible_matches_index]

        if not possible_matches.empty:
            
            intersecting_geoms = possible_matches[possible_matches.geometry.intersects(geometry_a['geometry'])]

            if not intersecting_geoms.empty:
                result_A.append(geometry_a)

    # Tạo GeoDataFrame từ kết quả
    selected_from_shp_a = gpd.GeoDataFrame(result_A, crs=shp_a.crs)

    sindex_a = shp_a.sindex

# Tìm các đa giác của A có ít nhất một phần giao với B sử dụng Spatial Index
    result_B = []

    for idx_b, geometry_b in shp_b.iterrows():
        possible_matches_index_b = list(sindex_a.intersection(geometry_b['geometry'].bounds))
        possible_matches_b = shp_a.iloc[possible_matches_index_b]

        if not possible_matches_b.empty:
            # Kiểm tra đa giác của A có giao với các đa giác của B hay không
            intersecting_geoms_b = possible_matches_b[possible_matches_b.geometry.intersects(geometry_b['geometry'])]

            if not intersecting_geoms_b.empty:
                result_B.append(geometry_b)

    # Tạo GeoDataFrame từ kết quả
    selected_from_shp_b = gpd.GeoDataFrame(result_B, crs=shp_a.crs)

    print('union')
    union_result = gpd.overlay(selected_from_shp_a, selected_from_shp_b, how='union')
    #dislove 
    print('dissloved')
    dissolved_result = union_result.dissolve()
    #multipart to single part
    print('explode')
    singlepart_gdf = dissolved_result.explode()
    singlepart_gdf.to_file(out)

    return out

if __name__ == '__main__':

    input_out_shp_multi = '/home/skymap/data/Bahrain_change/allqc/BAOCAO_25_3/TEST_kq_cac_epoch/Epoc_681/kq_multi/s1920/set1920/out_shp/merge'
    input_out_no_multi = '/home/skymap/data/Bahrain_change/allqc/BAOCAO_25_3/TEST_kq_cac_epoch/Epoc_681/kq_multi/s1920/set1920/out_shp/No_multil'
    path_dir_out = '/home/skymap/data/Bahrain_change/allqc/BAOCAO_25_3/TEST_kq_cac_epoch/Epoc_681/kq_multi/s1920/set1920/out_shp/inter_mul_no'

    for fp_shp_multi in glob.glob(os.path.join(input_out_shp_multi,'*.shp')):
        full_name = os.path.basename(fp_shp_multi)
        name, _ = os.path.splitext(full_name)
        path_nomulti = os.path.join(input_out_no_multi,f'{name}.shp')
        path_out = os.path.join(path_dir_out,f'{name}.shp')
        shp_multi = gpd.read_file(fp_shp_multi)
        shp_nomulti = gpd.read_file(path_nomulti)
        print('dang xu li ',fp_shp_multi)
        shp_process_new(shp_multi,shp_nomulti,path_out)


