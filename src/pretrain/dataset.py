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

# /work/.medsam/dataset
jsonlist1=r'/home/work/.medsam/dataset/cvpr/jsons/dataset_LUNA16_0.json'
datadir1=r'/home/work/.medsam/dataset/LUNA16'
datalist1 = load_decathlon_datalist(jsonlist1, False, "training", base_dir=datadir1)
print("Dataset 1 LUNA16 : number of data: {}".format(len(datalist1)))
new_datalist1 = []
for item in datalist1:
    item_dict = {"image": item["image"]}
    new_datalist1.append(item_dict)

jsonlist2=r'/home/work/.medsam/dataset/cvpr/jsons/dataset_TCIAcovid19_0.json'
datadir2 =r'/home/work/.medsam/dataset/TCIAcovid19'
datalist2 = load_decathlon_datalist(jsonlist2, False, "training", base_dir=datadir2)
print("Dataset 2 Covid19: number of data: {}".format(len(datalist2)))
new_datalist2 = []
for item in datalist2:
    item_dict = {"image": item["image"]}
    new_datalist2.append(item_dict)



jsonlist123=r'/home/work/.medsam/dataset/BTCV/dataset_0.json'
datadir123 =r'/home/work/.medsam/dataset/BTCV'
val_list = load_decathlon_datalist(jsonlist123, False, "all", base_dir=datadir123)
print("Validation: number of data: {}".format(len(val_list)))
new_datalist7= []
for item in val_list:
    item_dict = {"image": item["image"]}
    new_datalist7.append(item_dict)



jsonlist7=r'/home/work/.medsam/dataset/BTCV/dataset_0.json'
datadir7 =r'/home/work/.medsam/dataset/BTCV'
val_list = load_decathlon_datalist(jsonlist7, False, "test", base_dir=datadir7)
print("Validation: number of data: {}".format(len(val_list)))
val_datalist1 = []
for item in val_list:
    item_dict = {"image": item["image"]}
    val_datalist1.append(item_dict)


jsonlist8 =r'/home/work/.medsam/dataset/TCIAcovid19/dataset_TCIAcovid19_0.json'
val_list2 = load_decathlon_datalist(jsonlist2, False, "validation", base_dir=datadir2)
print("Validation Covid19: number of data: {}".format(len(val_list2)))
val_datalist2 = []
for item in val_list2:
    item_dict = {"image": item["image"]}
    val_datalist2.append(item_dict)

val = val_datalist1 + val_datalist2 # + val_datalist3 +val_datalist4
datalist =  new_datalist1 + new_datalist2 + new_datalist7 #+ new_datalist7 + new_datalist8 + new_datalist9 + new_datalist10 + new_datalist11
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