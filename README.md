# BioTac_slide_20_50
This repository contains tactile data and material description for texture classification, supplementary to the paper "Supervised  Autoencoder  Joint  Learning  on  Heterogeneous  TactileSensory  Data:  Improving  Material  Classification  Performance", R. Gao et al.

## Dataset description
This dataset concerns with 20 classes named with `mat + x`, where `x` is an index number. A snapshot of each corresponding material is shown below and the detailed description for each material can be found in `material_properties` text file by the name index or two-letter code.
![material_snaposhots](matx_collage.png "Snapshots of 20 materials")
<br/>
Each material constitutes a separate folder in which 50 samples are presented, named by `matx_n`, where `x` is the material index and `n` is the sample index. For each sample, we have a series of `.csv` files compiled from original `rosbag` data and one `.pt` file that was used as input data for BioTac model mentioned in the above paper. <br/>
* For `.csv` files, the postfix after sample name indicates the recorded ros topic: `bio`-- BioTac sensor reading, `cp`-- cartesian position, `cv`-- cartesian velocity, `cw`-- cartesian wrench, `jp`-- joint position, `jt`-- joint torque, `jv`-- joint velocity.
* For `.pt` file, it is extracted from `_bio.csv` file for each sample and formatted as a `400*44` tensor, where `400` is the sequence length and `44` is the number of the data fields, as shown in the header of `_bio.csv` file. It can be directly imported by `torch.load(filename)` and used as input data.

