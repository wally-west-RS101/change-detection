import geopandas as gpd

import os,glob

input_cd = '/home/skymap/data/Bahrain_change/allqc/BAOCAO_13_5/SHP'
input_shadow = '/home/skymap/data/Bahrain_change/allqc/BAOCAO_13_5/shp_shadow'
out_dir = '/home/skymap/data/Bahrain_change/allqc/BAOCAO_13_5/out_inter_vs_shadow'
for fp_cd in glob.glob(os.path.join(input_cd,'*.shp')):
    full_name = os.path.basename(fp_cd)
    name, _ = os.path.splitext(full_name)
    path_shp_b = os.path.join(input_shadow,f'{name}.shp')

# Đọc dữ liệu từ hai shapefile
    shp_a = gpd.read_file(fp_cd)
    shp_b = gpd.read_file(path_shp_b)
    print('dang xu li ',fp_cd , path_shp_b)

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
    result = gpd.overlay(shp_a, selected_from_shp_a, how='difference')
    out = os.path.join(out_dir,f'{name}.shp')
    print('luuuu',out)
    # Lưu kết quả vào một shapefile mới
    result.to_file(out)
    print('done')