import geopandas as gpd

shapefile1_path = "/home/skymap/data/Bahrain_change/allqc/BAOCAO_26_2/out_kq_all/set2122/out_shp/merge/SET_2122.shp"
shapefile2_path = "/home/skymap/data/Bahrain_change/allqc/BAOCAO_26_2/KQ_nomulti/SET_2122.shp"


gdf1 = gpd.read_file(shapefile1_path)
gdf2 = gpd.read_file(shapefile2_path)


intersection_result = gpd.overlay(gdf2, gdf1, how='intersection')
# intersection_result = gdf1.intersection(gdf2)

output_path = "/home/skymap/data/Bahrain_change/allqc/BAOCAO_26_2/intersec/SET_2122.shp"

intersection_result.to_file(output_path)

print(f"Kết quả đã được ghi vào {output_path}")

