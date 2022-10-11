# Use this script to run tasks in parallel. Change config path and count.

export OMP_NUM_THREADS=1

parallel --eta --ungroup --jobs 5 python main_fa.py -c configs/tc_mountaincar_10_{1}.yaml -e tc_mountaincar_{1} ::: $(seq 1 5)