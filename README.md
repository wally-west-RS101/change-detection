# Folder structure

        - code_processing
                - all_process
                - post_process        
        - code_train
                - model_changedetection
                - model_classify
                - model_segmentation

# Pipeline 

        Step 1: Detect change detection with two model changeformer (using optimizer adamw and sgd) combine model 
                combined with the classification model VIT
                link paper model changeformer : https://arxiv.org/abs/2201.01293
                link github model changeformer : https://github.com/wgcban/ChangeFormer.git
                link paper model VIT : https://arxiv.org/abs/2010.11929v2
                link github model VIT : https://github.com/lucidrains/vit-pytorch.git

        Step 2: Detects shadows and water surfaces to perform false detection deletions using model segmentation U2net
        
        Step 3 : Postprocess 
                - procesing raster and convert to vecto
                - processing vecto

# Training
## Training model Changeformer

        - Setting 
            1. Setting path datat train in dataconfig.py
            2. Setting confing in main_cd.py
                - project_name : name folder checkpoint
                - dataset/dataset_name : using default (CDDataset/LEVIR)
                - batch_size : batch_size for training
                - img_size : size input model (256)
                - n_class : numbers class (4)
                - embed_dim : 64
                - pretrain : path pretrain model
                - multi_scale_train : True
                - multi_scale_infer : False
                - net_G : model using ChangeFormerV6
            3. if you want to use agumentation , edit dataset/CD_dataset.py
            4. Run python main_cd.py
        
        - Data training
            Change detection data set with pixel-level binary labels；
            ├─A
            ├─B
            ├─label
            └─list
        
            A: images of t1 phase;

            B:images of t2 phase;

            label: label maps;

            list: contains train.txt, val.txt and test.txt, each file records the image names (XXX.tif) in the change detection dataset.

## Training Classification model VIT transformer

            - Training using model_classify/training_change.ipynb

            - Setting
                batch_size = numbers os batch size
                epochs = numbers of epochs
                lr = 3e-5
                gamma = 0.7
                seed = 42

            - config model :

                efficient_transformer = Linformer(
                dim=128,
                seq_len=49+1,  # 7x7 patches + 1 cls-token
                depth=12,
                heads=8,
                k=64
                )

                model = ViT(
                    dim=128,
                    image_size=224,
                    patch_size=32,
                    num_classes=2,
                    transformer=efficient_transformer,
                    channels=8,
                ).to(device)
            
            - Data training
                Classification model dataset :
                ├─DATA─ change_image1.png
                    ├─ nochange_image2.png
            
                "change" and "nochange" is label

## Training model segmentation U2net (model detect water and shadow)

            - Training model U2net using file u2net/training_tif_viz.py
            - config model:
                - batch_size : batch size model
                - load_weights : load weight path model pretrain
                - TRAIN_EPOCHS : numbers epoch

            - Data training
                Segmentation model dataset :
                ├── train
                │   ├── image
                │   │   ├── file .tif
                │   ├── label
                │   │   ├── file .tif
                ├── val
                │   ├── image
                │   │   ├── file .tif
                │   ├── label
                │   │   ├── file .tif

## Link pretrain weights
        link : /home/ml/ml_data/Dao_WS/weight_changedetection
## DATA training
        link : /home/ml/ml_data/Dao_WS/all_data_training

# Post process
## All_process

                - The end-to-end subtraction process of the change detection problem with the input being 
                2 folders containing pairs of images of different times.
                 The result is a folder containing the resulting raster and vector

                - Config :

                    n_class = numbers class of model changeformer (4)
                    check_point_path_sgd = path model changeformer using sgd optimizer
                    check_point_path_adam = path model changeformer using adamw optimizer
                    model_path_classify = path model classification
                    embeddim = 64
                    num_bands = 4
                    path_model_shadow = path model detect shadow
                    path_model_water = path_model_detect water
                    kenel_size_mopho = kernel size for mophology (11)
                    min_area_shp = minimum area to get polygon
                    min_area_holes = area delete holes
                    img_size = size model (If using models with different inputs, other img_size variables need to be created)

                - Run
                    - run file main_all.py
                
## Post_pocess
                - calculate_f1_precision_recall.py : Calculate f1, precision, recall
                - delete_class_with_raster.py : Delete class with raster
                - delete_polygon_with_AOI.py : Delete polygon with AOI 
                - intersec_two_shp.py : Intersec two shapefile 
                - raster_to_vecto : Convert raster to vecto
                - select_polygon_intersec_two_shp.py : Get the union of two intersecting polygons
