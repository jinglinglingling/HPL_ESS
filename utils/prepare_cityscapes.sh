path_to_zipdataset="/mnt/workspace/dingyiming/DataBase/Cityscapes"
path_to_dataset="./data/cityscapes"

# 扩展训练数据
# unzip  ${path_to_zipdataset}"/leftImg8bit_trainextra.zip" -d ${path_to_dataset}
unzip  ${path_to_zipdataset}"/gtFine_trainvaltest.zip" -d ${path_to_dataset}
unzip  ${path_to_zipdataset}"/leftImg8bit_trainvaltest.zip" -d ${path_to_dataset}




