path_to_zipdataset="/mnt/workspace/dingyiming/DataBase/GTA5/"
path_to_dataset="./data/gta/"

for file_name in `ls ${path_to_zipdataset} | grep 'images'`
do
        unzip ${path_to_zipdataset}${file_name} -d ${path_to_dataset}
done

for file_name in `ls ${path_to_zipdataset} | grep 'labels'`
do
        unzip ${path_to_zipdataset}${file_name} -d ${path_to_dataset}
done
