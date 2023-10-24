# Use this script to verify BanditPAM is multithreaded
# Should complete in <= 3 seconds
import time
import numpy as np
import banditpam

banditpam.set_num_threads(1)

X = np.loadtxt("data/MNIST_1k.csv")
kmed = banditpam.KMedoids(n_medoids=5, algorithm="BanditPAM")
start = time.time()
kmed.fit(X, "L2")
print(time.time() - start, "seconds")
print("Number of SWAP steps:", kmed.steps)
print(kmed.medoids)

print("\n\nStatistics:")
print("Build distance comps:", kmed.build_distance_computations)
print("Swap distance comps:", kmed.swap_distance_computations)
print("Misc distance comps:", kmed.misc_distance_computations)
print("Build + Swap distance comps:", kmed.getDistanceComputations(False), kmed.build_distance_computations + kmed.swap_distance_computations)
print("Total distance comps:", kmed.getDistanceComputations(True), kmed.build_distance_computations + kmed.swap_distance_computations + kmed.misc_distance_computations)

# print(kmed.getDistanceComputations())  # Unsupported, must provide arg
print("Number of Steps:", kmed.steps)
print("Total Build time:", kmed.total_build_time)
print("Total Swap time:", kmed.total_swap_time)
print("Time per swap:", kmed.time_per_swap, kmed.total_swap_time / kmed.steps)
print("Total time:", kmed.total_time)

print("Build loss:", kmed.build_loss)
print("Final loss:", kmed.average_loss)
print("Loss trajectory:", kmed.losses)

print("Build medoids:", kmed.build_medoids)
print("Final medoids:", kmed.medoids)
