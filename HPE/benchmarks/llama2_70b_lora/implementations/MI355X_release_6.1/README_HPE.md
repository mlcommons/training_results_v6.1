# Llama 2 70B LoRA, HPE XD685 (8x MI355X)

Code: AMD Primus v26.7.0 (`2631e68d`), recipe `examples/mlperf/llama2_70b`, run in
`rocm/primus:v26.7` with the Primus checkout mounted over `/workspace/Primus`.
HPE changes are in `primus_v26.7.0_log61.patch` (also shipped as modified files under
`primus_v26.7.0/`): MLPerf 6.1 precision log fields and submitter fields, schedule end
logged as `opt_learning_rate_training_steps` (unchanged value 550), per-trial seeds from
`/dev/urandom`.

Launch (one Slurm job, 20 runs in one group):

    sbatch -w lodin2 --time=6:00:00 \
      --comment="docker;sysctl file=/home/hpebench/sysctl/zone_reclaim0.conf;thp=madvise" \
      --export=ALL,BENCH=llama2_70b,TAG=g20-zr0-thpmadvise,PRIMUS_HOST=<patched Primus v26.7.0> \
      run_primus_llama.sbatch 20

The `--comment` keywords make the cluster's Slurm prolog apply, for the job only:
`vm.zone_reclaim_mode=0` (file `zone_reclaim0.conf`) and transparent huge pages
`enabled=madvise`, `defrag=madvise`.

Scoring: 20 runs, ordered by start time; the 10 consecutive runs closest to the median
(runs 5-14 of 20) are the submitted `result_1..result_10`. All 20 logs are kept and
available on request.
