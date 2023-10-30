# BanditPAM++: Faster $k$-Medoids Clustering

This repository contains the code to reproduce all the experimental results from "BanditPAM++: Faster $k$-medoids Clustering"
by Mo Tiwari, Donghyun (Luke) Lee*, Ryan Kang*, Sebastian Thrun, Chris Piech, Ilan Shomorony, and Martin Jinye Zhang, 
published in Advances in Neural Information Processing Systems (NeurIPS) 2023.

This repository is a fork of the original [BanditPAM](https://github.com/motiwari/BanditPAM) repository, with the appropriate
changes to reproduce the experiments from the BanditPAM++ paper. Note that the BanditPAM++ algorithm has already been
incorporated into the original BanditPAM repository.

If you use this code, please cite:
    
```bibtex
@inproceedings{BanditPAMpp,
  title={BanditPAM++: Faster k-medoids Clustering},
  author={Tiwari, Mo and Lee, Donghyun, and Kang, Ryan and Thrun, Sebastian and Piech, Chris and Shomorony, Ilan and Zhang, Martin Jinye},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}

@inproceedings{BanditPAM,
  title={BanditPAM: Almost Linear Time k-medoids Clustering via Multi-Armed Bandits},
  author={Tiwari, Mo and Zhang, Martin J and Mayclin, James and Thrun, Sebastian and Piech, Chris and Shomorony, Ilan},
  booktitle={Advances in Neural Information Processing Systems},
  pages={368--374},
  year={2020}
}

```

## Instructions to reproduce the experiments

Note that reproducing all the experiments is only supported on Mac OSX. 
To reproduce all results, simply run:

```bash
./repro_script.sh
```

This will download all the data, run the experiments, save all the logfiles to `experiments/logs/`, save the plots in 
`experiments/figures/`, and print the results for the tables to the terminal. Note that running the experiments can take
a very, very long time (over a week). For the paper, we ran the experiments on AWS using c5.4xlarge instances with 64GB
of disk space.