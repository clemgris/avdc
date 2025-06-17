def update_feat_transform(policy_data_config, transforms_dict):
    if "dino" in policy_data_config.datamodule.lang_dataset.diffuse_on:
        for section in ["train", "val"]:
            if hasattr(transforms_dict, section) and hasattr(
                transforms_dict[section], "rgb_static"
            ):
                for transform in transforms_dict[section].rgb_static:
                    if (
                        "_target_" in transform
                        and transform["_target_"] == "torchvision.transforms.Resize"
                    ):
                        print(
                            f"Resize {section} to",
                            policy_data_config.datamodule.lang_dataset.feat_patch_size
                            * 14,
                        )
                        transform["size"] = (
                            policy_data_config.datamodule.lang_dataset.feat_patch_size
                            * 14
                        )
                    elif (
                        "_target_" in transform
                        and transform["_target_"] == "torchvision.transforms.CenterCrop"
                    ):
                        print(
                            f"CenterCrop {section} to",
                            policy_data_config.datamodule.lang_dataset.feat_patch_size
                            * 14,
                        )
                        transform["size"] = (
                            policy_data_config.datamodule.lang_dataset.feat_patch_size
                            * 14
                        )
    elif "r3m" in policy_data_config.datamodule.lang_dataset.diffuse_on:
        for section in ["train", "val"]:
            if hasattr(transforms_dict, section) and hasattr(
                transforms_dict[section], "rgb_static"
            ):
                for transform in transforms_dict[section].rgb_static:
                    if (
                        "_target_" in transform
                        and transform["_target_"] == "torchvision.transforms.Resize"
                    ):
                        print(
                            f"Resize {section} to",
                            policy_data_config.datamodule.lang_dataset.feat_patch_size
                            * 32,
                        )
                        transform["size"] = (
                            policy_data_config.datamodule.lang_dataset.feat_patch_size
                            * 32
                        )
                    elif (
                        "_target_" in transform
                        and transform["_target_"] == "torchvision.transforms.CenterCrop"
                    ):
                        print(
                            f"CenterCrop {section} to",
                            policy_data_config.datamodule.lang_dataset.feat_patch_size
                            * 32,
                        )
                        transform["size"] = (
                            policy_data_config.datamodule.lang_dataset.feat_patch_size
                            * 32
                        )
    return transforms_dict
