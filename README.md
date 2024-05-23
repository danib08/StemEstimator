# StemEstimator

## Description
This project aims to calculate the different radii values of tree stems in a forest plot represented by a point cloud.

The trees are segmented and filtered from the rest of the point cloud, and then most of its leaves and branches are removed. Ellipses are fitted along the stems in order to approximate its radii. Finally, the results are displayed to the user.

## Installation
Anaconda is highly recommended in order to install `pclpy`.

Run `conda env create -f environment.yml` and then `conda activate stemestimator`. 

Open3D is not distributed through Anaconda and needs to be installed through `pip`:

Run `conda install pip` and then `pip install -r requirements.txt`

## Usage
Run `app.py` and upload a point cloud file. Currently only `.pcd` and `.xyz` files are supported. The latter requires the x-y-z values to be on columns 2-3-4 respectively.

When the processing is done, the stems and their ellipses can be visualized as a whole, or a specific tree may be selected in order to show the radius results.

## Contributing
Technical modifications to the point cloud processing can be done in the `tree_tool.py` module. The `TreeTool` class methods are leveraged by the `manager.py` module.

## Credits
This project is based on the incredible work by [1].

[1] O. Montoya, O. Icasio-Hernández, and J. Salas, “TreeTool: A tool for detecting trees and estimating their DBH using Forest Point Clouds,” SoftwareX, vol. 16, Nov. 2021. doi:10.1016/j.softx.2021.100889 