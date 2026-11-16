#!/usr/bin/env bash
                                                                      
 
                                                                            
                                                                          
                                                                    
                                                           
                                                          
                                                         
 
        
                                                    
                                                                       
 
      
                                                                
                                                                                     
                                                                            
                                                                
set -euo pipefail

DATASET="${DATASET:-yambda-5b}"
: "${DATADIR:?Set DATADIR to the destination data root}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

echo "[download_dataset] dataset=${DATASET} data-path=${DATADIR}"
mkdir -p "${DATADIR}"

python3 -m generative_recommenders.dlrm_v4.preprocess_public_data \
    --dataset "${DATASET}" \
    --data-path "${DATADIR}"

echo "[download_dataset] done. Preprocessed layout under ${DATADIR}:"
echo "  raw/5b/multi_event.parquet"
echo "  shared_metadata/{artist,album}_item_mapping.parquet, embeddings.parquet"
echo "  processed_5b/{train_sessions,test_events,session_index}.parquet"
echo "  processed_5b/item_popularity.npy, processed_5b/split_meta.json"
echo "[download_dataset] verify integrity with: ./verify_dataset.sh"
