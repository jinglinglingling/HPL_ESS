path_to_zipdataset="/mnt/workspace/dingyiming/DataBase/DSEC"
path_to_dataset="./data/DSEC_Semantic"

if [ -d ${path_to_dataset} ]; then
    echo ${path_to_dataset}"   exist！"
else
    mkdir data
    mkdir ${path_to_dataset}
fi

if [ -d ${path_to_dataset}"/gt_fine/" ]; then
    echo ${path_to_dataset}"/gt_fine/""   exist！"
else
    mkdir ${path_to_dataset}"/gt_fine/"
    unzip ${path_to_zipdataset}"/train_semantic_segmentation.zip" -d ${path_to_dataset}"/gt_fine/"
    unzip ${path_to_zipdataset}"/test_semantic_segmentation.zip" -d ${path_to_dataset}"/gt_fine/"

    # rm -rf ${path_to_dataset}"/gt_fine/train/labels.py"
    # for file_name in `ls ${path_to_dataset}"/gt_fine/train"`
    # do
    #     for file_name_2 in `ls ${path_to_dataset}"/gt_fine/train/"${file_name}"/19classes/"`
    #     do
    #     mv ${path_to_dataset}"/gt_fine/train/"${file_name}"/19classes/"${file_name_2} ${path_to_dataset}"/gt_fine/train/"${file_name}
    #     done
    #     echo ${path_to_dataset}"/gt_fine/train/"${file_name}"/19classes/*""----->"${path_to_dataset}"/gt_fine/train/"${file_name}
    # done

    # for file_name in `ls ${path_to_dataset}"/gt_fine/train"`
    # do
    #     rm -rf ${path_to_dataset}"/gt_fine/train/"${file_name}"/11classes"
    #     rm -rf ${path_to_dataset}"/gt_fine/train/"${file_name}"/19classes"
    # done

    # rm -rf ${path_to_dataset}"/gt_fine/test/labels.py"
    # for file_name in `ls ${path_to_dataset}"/gt_fine/test"`
    # do
    #     for file_name_2 in `ls ${path_to_dataset}"/gt_fine/test/"${file_name}"/19classes/"`
    #     do
    #     mv ${path_to_dataset}"/gt_fine/test/"${file_name}"/19classes/"${file_name_2} ${path_to_dataset}"/gt_fine/test/"${file_name}
    #     done
    #     echo ${path_to_dataset}"/gt_fine/test/"${file_name}"/19classes/*""----->"${path_to_dataset}"/gt_fine/test/"${file_name}
    # done

    # for file_name in `ls ${path_to_dataset}"/gt_fine/test"`
    # do
    #     rm -rf ${path_to_dataset}"/gt_fine/test/"${file_name}"/11classes"
    #     rm -rf ${path_to_dataset}"/gt_fine/test/"${file_name}"/19classes"
    # done
fi

if [ -d ${path_to_dataset}"/events/train" ]; then
    echo ${path_to_dataset}"/events/train""   exist！"
else
    mkdir ${path_to_dataset}"/events"
    mkdir ${path_to_dataset}"/events/train"
    unzip ${path_to_zipdataset}"/train_events.zip" -d ${path_to_dataset}"/events/train"
fi

if [ -d ${path_to_dataset}"/events/test" ]; then
    echo ${path_to_dataset}"/events/test""   exist！"
else
    mkdir ${path_to_dataset}"/events/test"
    unzip ${path_to_zipdataset}"/test_events.zip" -d ${path_to_dataset}"/events/test"
fi

if [ -d ${path_to_dataset}"/images/train" ]; then
    echo ${path_to_dataset}"/images/train""   exist！"
else
    mkdir ${path_to_dataset}"/images"
    mkdir ${path_to_dataset}"/images/train"
    unzip ${path_to_zipdataset}"/train_images.zip" -d ${path_to_dataset}"/images/train"

    for file_name in `ls ${path_to_dataset}"/images/train"`
    do
        mv ${path_to_dataset}"/images/train/"${file_name}"/images/timestamps.txt" ${path_to_dataset}"/images/train/"${file_name}
        mv ${path_to_dataset}"/images/train/"${file_name}"/images/left/exposure_timestamps.txt" ${path_to_dataset}"/images/train/"${file_name}
        for file_name_2 in `ls ${path_to_dataset}"/images/train/"${file_name}"/images/left/rectified/"`
        do
        mv ${path_to_dataset}"/images/train/"${file_name}"/images/left/rectified/"${file_name_2} ${path_to_dataset}"/images/train/"${file_name}
        done
        echo ${path_to_dataset}"/images/train/"${file_name}"/images/left/rectified/*""----->"${path_to_dataset}"/images/train/"${file_name}
    done

    for file_name in `ls ${path_to_dataset}"/images/train"`
    do
        rm -rf ${path_to_dataset}"/images/train/"${file_name}"/images"
    done
fi

if [ -d ${path_to_dataset}"/images/test" ]; then
    echo ${path_to_dataset}"/images/test""   exist！"
else
    mkdir ${path_to_dataset}"/images/test"
    unzip ${path_to_zipdataset}"/test_images.zip" -d ${path_to_dataset}"/images/test"

    for file_name in `ls ${path_to_dataset}"/images/test"`
    do
    mv ${path_to_dataset}"/images/test/"${file_name}"/images/timestamps.txt" ${path_to_dataset}"/images/test/"${file_name}
    mv ${path_to_dataset}"/images/test/"${file_name}"/images/left/exposure_timestamps.txt" ${path_to_dataset}"/images/test/"${file_name}
    for file_name_2 in `ls ${path_to_dataset}"/images/test/"${file_name}"/images/left/rectified/"`
    do
    mv ${path_to_dataset}"/images/test/"${file_name}"/images/left/rectified/"${file_name_2} ${path_to_dataset}"/images/test/"${file_name}
    done
    echo ${path_to_dataset}"/images/test/"${file_name}"/images/left/rectified/*""----->"${path_to_dataset}"/images/test/"${file_name}
    done

    for file_name in `ls ${path_to_dataset}"/images/test"`
    do
    rm -rf ${path_to_dataset}"/images/test/"${file_name}"/images"
    done
fi

echo "DSEC has been prepared at"${path_to_dataset}




