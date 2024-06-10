import tensorflow as tf
from models.import_module import Model_U2Net,Model_UNet3plus

my_model = Model_U2Net(512,6)


my_model.load_weights(r'/home/skymap/data/Bahrain_change/tftotorch/u2net_512_bd_bahrain_update_12.h5')
my_model.save(r'/home/skymap/data/Bahrain_change/tftotorch/u2net_512_bd_bahrain_update_12_model.h5')

# 20220826_065957_ssc3_u0001_visual
# 20220821_070453_ssc12_u0001_visual
# 20220820_103902_ssc6_u0001_visual
# 20220813_070157_ssc12_u0002_visual

# sua
# 20220821_070453_ssc12_u0002_visual
# 20220819_102529_ssc8_u0002_visual
# 20220818_073620_ssc1_u0002_visual
# 20220814_103841_ssc10_u0002_visual
# 20220814_103841_ssc10_u0001_visual
# 20220807_064607_ssc4_u0002_visual
# 20220812_070039_ssc4_u0002_visual
# 20220813_070157_ssc12_u0001_visual
