# Mobilkit+
This project aims to streamline data processing operations for multiple projects in the Urban Mobility, Networks and Intelligence (UMNI) lab at Purdue University, led by [Dr. Satish Ukkusuri](http://www.satishukkusuri.com/). The name `mobilkit` is the same as [github.com/mindearth/mobilkit](github.com/mindearth/mobilkit), also co-developed by UMNI lab, but this package does more and different than the original `mobilkit`.

<!-- ![Mobilkit framework diagram](./_common/fig/Mobilkit.png) -->

## Files and resources:
- **[Framework in Figma](https://www.figma.com/file/LqnQC54G4w6CaDwsGZExXU/Mobil?node-id=0%3A1&t=kH061lIHBTjiACSy-1)**
- **Mobilkit Extension.pptx**: Main presentation to record our weekly/biweekly updates.
- **[Notion document](https://emphasent.notion.site/Mobilkit-aa39edb3dd77487aac1768671a3a75ee)**: For documenting ideas and content details such as codebase description.
<!-- # - [Data and outputs (Shagun) (Google Slides)](https://docs.google.com/presentation/d/1tITgL1qcZMS7B1LDvlpn9QK8V123QDNhyCbJwtbO1Ds/edit#slide=id.p)
# - **`mobilkit`**: Source code of the current `mobilkit` library cloned from the [Github repository](https://github.com/mindearth/mobilkit).
# - **`mobilkitplus`**: Package structure of the proposed toolkit. The main modules are in the folder `mobil`. -->

## Installation
The current version of `mobilkit` uses `pyspark` for which it requires Python 3 version 3.9 or earlier.
It is recommended to install this package in a new virtual environment. In `conda`, this may be done as:
```bash
conda create -n mk python=3.9.7
conda activate mk
```
Then, it can be installed using `pip` from [PyPi](https://pypi.org/project/pip/):
```bash
pip install git+https://rvanxer@github.com/rvanxer/mk.git
```
