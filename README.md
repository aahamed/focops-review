# focops-review
Repository containing experiments to review the FOCOPS algorithm

## Changes

* Add code to track additional statistics such as standard deviation of cost, speed per timestep, average length of episode etc.
* Add support for discrete action spaces
* Add support for new safety environments from [here](https://github.com/jemaw/gym-safety)
* Add code for plotting results
* Add support to resume training from checkpoint

## Usage

### Focops
You can run the FOCOPS algorithm using the following command:
```
python focops_main.py --constraint=other --max-iter-num=100 --exp=<EXP_NAME> --env-id=GridNav-v0 --max-eps-len=200
```
You can set the cost thresholds for an environment in the environments.py file.
Results are saved in the `focops_results/<env-id>/<exp>/` directory.

### Plotting
You can plot the results using the following command:
```
 python plot.py --log_dir=focops_results/GridNav-v0/<exp>
```
Plots are stored in the log_dir.
