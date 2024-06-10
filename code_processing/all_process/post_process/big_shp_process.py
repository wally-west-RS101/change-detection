import geopandas as gpd

# def remove_water(shp_result, AOI_watwer):
  
#     shpfile_multiple_polygons = gpd.read_file(shp_result)
#     shpfile_single_polygon = gpd.read_file(AOI_watwer)
#     single_polygon_geometry = shpfile_single_polygon.geometry.iloc[0]
#     selected_polygons = shpfile_multiple_polygons[shpfile_multiple_polygons.geometry.intersects(single_polygon_geometry)]   
#     selected_polygons.to_file(shp_result, driver='ESRI Shapefile')

#     return shp_result



shapefile1_path = "/home/skymap/data/Bahrain_change/allqc/BAOCAO_26_2/out_kq_all/set2122/out_shp/merge/SET_2122.shp"
shapefile2_path = "/home/skymap/data/Bahrain_change/allqc/BAOCAO_26_2/KQ_nomulti/SET_2122.shp"


gdf1 = gpd.read_file(shapefile1_path)
gdf2 = gpd.read_file(shapefile2_path)


intersection_result = gpd.overlay(gdf2, gdf1, how='intersection')
# intersection_result = gdf1.intersection(gdf2)

output_path = "/home/skymap/data/Bahrain_change/allqc/BAOCAO_26_2/intersec/SET_2122.shp"

intersection_result.to_file(output_path)

print(f"Kết quả đã được ghi vào {output_path}")



# import geopandas as gpd
# from shapely.geometry import Polygon


# shp_a = gpd.read_file('/home/skymap/data/Bahrain_change/out_chang_fmer_test_s2223/merge/T22_23_changeformer.shp')
# shp_b = gpd.read_file('/home/skymap/data/Bahrain_change/out_chang_fmer_test_s2223/intersec/inter.shp')


# selected_polygons = []

# for index, polygon_a in shp_a.iterrows():
   
#     intersects_with_b = any(polygon_a['geometry'].intersects(polygon_b) for _, polygon_b in shp_b.iterrows())
    

#     if intersects_with_b:
#         selected_polygons.append(polygon_a)

# result = gpd.GeoDataFrame(selected_polygons, crs=shp_a.crs)


# result.to_file('/home/skymap/data/Bahrain_change/out_chang_fmer_test_s2223/testtt/a.shp')