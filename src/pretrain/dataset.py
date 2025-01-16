import os
from Pretraining.pretrain import args
from monai.data import CacheDataset, DataLoader, Dataset,load_decathlon_datalist
from monai.data import (
    DataLoader,
    load_decathlon_datalist,

)
from monai.transforms import (
EnsureChannelFirstd,
    AsDiscrete,
    Compose,
    LoadImaged,
    Orientationd,
    RandSpatialCropd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToTensord,
    EnsureTyped
)



datadir1=args.luna_path
jsonlist1=os.path.join(datadir1,"LUNA16.json")
datalist1 = load_decathlon_datalist(jsonlist1, False, "training", base_dir=datadir1)
print("Dataset 1 LUNA16 : number of data: {}".format(len(datalist1)))
new_datalist1 = []
for item in datalist1:
    item_dict = {"image": item["image"]}
    new_datalist1.append(item_dict)

datadir2=args.btcv_path
jsonlist2=os.path.join(datadir1,"dataset_0.json")
datalist2 = load_decathlon_datalist(jsonlist2, False, "all", base_dir=datadir2)
print("Dataset 2 BTCV: number of data: {}".format(len(datalist2)))
new_datalist2 = []
for item in datalist2:
    item_dict = {"image": item["image"]}
    new_datalist2.append(item_dict)

datadir3=args.covid_path
jsonlist3=os.path.join(datadir1,"Covid19.json")
datalist3 = load_decathlon_datalist(jsonlist2, False, "training", base_dir=datadir3)
print("Dataset 2 Covid19: number of data: {}".format(len(datalist3)))
new_datalist3 = []
for item in datalist3:
    item_dict = {"image": item["image"]}
    new_datalist3.append(item_dict)


val_list = load_decathlon_datalist(jsonlist2, False, "test", base_dir=datadir2)
print("Validation BTCV: number of data: {}".format(len(val_list)))
val_datalist1 = []
for item in val_list:
    item_dict = {"image": item["image"]}
    val_datalist1.append(item_dict)

val_list2 = load_decathlon_datalist(jsonlist3, False, "validation", base_dir=datadir3)
print("Validation Covid19: number of data: {}".format(len(val_list2)))
val_datalist2 = []
for item in val_list2:
    item_dict = {"image": item["image"]}
    val_datalist2.append(item_dict)

val = val_datalist1 + val_datalist2 
datalist =  new_datalist1 + new_datalist2 + new_datalist3 
print("Dataset all training: number of data: {}".format(len(datalist)))


train_transforms = Compose(
            [
                LoadImaged(keys=["image"],image_only=True),
                EnsureChannelFirstd(keys=["image"]),
                Spacingd(keys=["image"], pixdim=(args.space_x,args.space_y,args.space_z), mode=("bilinear")),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRanged(keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True),
                SpatialPadd(keys="image", spatial_size=[args.roi_x,args.roi_y,args.roi_z]),
                RandSpatialCropd(roi_size=[args.roi_x,args.roi_y,args.roi_z], keys=["image"], random_size=False, random_center=True),
                ToTensord(keys=["image"]),
                EnsureTyped(keys=["image"], device='cpu', track_meta=False),
            ]
        )

train_ds = Dataset(data=datalist, transform=train_transforms)
train_loader = DataLoader(train_ds, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True,pin_memory=True,pin_memory_device='cuda',persistent_workers=args.num_workers)#,prefetch_factor=69)
val_ds = Dataset(data=val, transform=train_transforms)
val_loader= DataLoader(val_ds, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,pin_memory=True,pin_memory_device='cuda',persistent_workers=args.num_workers)