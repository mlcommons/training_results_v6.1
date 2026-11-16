# Krai llama2_70b_lora on PowerEdge_XE9680_8xMI300X (Dell PowerEdge XE9680, 8x AMD Instinct MI300X 192 GB)

1. `./setup.sh`: builds `krai4ai/mlperf:training_v6.1-llama2_70b_lora-mi300x` from `../nemo/Dockerfile` (AMD's
   public v5.1 container with the changed recipe files, Transformer Engine's two one-line changes and the patch
   modules built in) and applies AMD's host tuning (`../nemo/runtime_tunables.sh`, root).
2. `HF_TOKEN=... ./init_datasets.sh`: dataset and model preparation as in AMD's v5.1 implementation, into
   `$DATADIR` (default `/data/mlperf_llama2`).
3. `NEXP=10 ./run_and_time.sh`: ten runs to eval loss <= 0.925 through AMD's unchanged `run_with_docker.sh`, logs
   `$LOGDIR/result_<i>.txt`. All settings are in `../nemo/config_MI300X_1x8x1.sh`; the changes vs AMD's
   implementation are listed in `../nemo/README.md`.

Host state used for the results: performance CPU governor, C1E disabled, transparent huge pages always, NUMA
balancing / NMI watchdog / ASLR off (AMD's runtime_tunables.sh), GPU power cap 750 W (default).
