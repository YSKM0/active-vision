NB_DATASET_PATH=/local/home/hanwliu/table
ALL_TRANSFORM=${NB_DATASET_PATH}/nerfstudio/transforms.json 

MODE=train
ALL_TRAIN_TRANSFORM=${NB_DATASET_PATH}/dataset/transforms_train.json 

DIST_TYPE=gcd
VLM_DIST_TYPE=cos
EMBEDDINGS_PATH=${colmap_log}/${VLM_EMBEDDING_FILE_PATH}

REP=1
EXP=${EXPERIMENT_NAME}
RESULT_PATH=${NB_DATASET_PATH}/dataset/train/nerfdirector

CHECKPOINT_DIR=${RESULT_PATH}/${EXP}/${REP}
BASE_IMAGE_DIR=${NB_DATASET_PATH}/${PATH_TO_IMAGES}
COLMAP_LOG=${NB_DATASET_PATH}/nerfstudio/colmap/txt/images.txt
VIZ_DIR=${RESULT_PATH}/${EXP}/viz

ALPHA=0.6

# Running
python ${PATH_TO_active_vision_select.py} --rep ${REP} \
                        --sampling vlm \
                        --real_world_data \
                        --all_train_transform ${ALL_TRAIN_TRANSFORM} \
                        --checkpoint_dir ${CHECKPOINT_DIR} \
                        --base_image_dir ${BASE_IMAGE_DIR} \
                        --all_transform ${ALL_TRANSFORM} \
                        --dist_type ${DIST_TYPE} \
                        --vlm_dist_type ${VLM_DIST_TYPE} \
                        --embeddings_path ${EMBEDDINGS_PATH} \
                        --use_val

                        # --viz_dir ${VIZ_DIR} \
                        # --enable_photo_dist \
                        # --colmap_log ${COLMAP_LOG} \
                        # --alpha ${ALPHA} \
                        # --copy_images \