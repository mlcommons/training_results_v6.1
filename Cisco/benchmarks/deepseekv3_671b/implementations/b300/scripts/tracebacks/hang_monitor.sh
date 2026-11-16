#!/usr/bin/env bash
                                                                                  
                                                                                   
                                            
 
                                                                        
                                                                                        
                                                                           
 
                                                                                   
                                                                                     
                                         
 
                                                                                 
                                                                                
                                                                                   
 
      
                                                                                 
                                                                         
                                           
                                                                                    
                                                                                      
                                                                  
                                                           
                                                                                 
                                                                                       
                                                                                 
                                                               
                                                                                       
                                                                                        

                                                                                 
timeleft_to_seconds() {
    local timeleft="$1" days=0 parts total
    case "$timeleft" in
        *UNLIMITED*|*INVALID*|"") echo ""; return 0 ;;
    esac
    if [[ "$timeleft" == *-* ]]; then
        days="${timeleft%%-*}"
        timeleft="${timeleft#*-}"
    fi
    IFS=':' read -r -a parts <<< "$timeleft"
    case "${#parts[@]}" in
        3) total=$(( 10#${parts[0]}*3600 + 10#${parts[1]}*60 + 10#${parts[2]} )) ;;
        2) total=$(( 10#${parts[0]}*60 + 10#${parts[1]} )) ;;
        *) echo ""; return 0 ;;
    esac
    echo $(( total + 10#${days}*86400 ))
}

                                                                         
_hang_monitor_log_size() {
    local glob="$1" total=0 f sz
    for f in $glob; do
        [ -f "$f" ] || continue
        sz="$(stat -c %s "$f" 2>/dev/null || echo 0)"
        total=$(( total + sz ))
    done
    echo "$total"
}

hang_monitor() {
    : "${HANG_MONITOR_TIMEOUT:?HANG_MONITOR_TIMEOUT not set}"
    : "${HANG_MONITOR_EXEC_CMD:?HANG_MONITOR_EXEC_CMD not set}"
    : "${SLURM_JOBID:?SLURM_JOBID not set}"
                                                                                    
                                                                                
                                                                                     
                                                                              
    : "${HANG_MONITOR_STALL_MINUTES:=0}"
    : "${HANG_MONITOR_POLL_SECONDS:=60}"
    : "${HANG_MONITOR_LOG_GLOB:=${LOG_FILE_BASE:-/dev/null}_*.log}"
    : "${HANG_MONITOR_MAX_DUMPS:=1}"
    : "${HANG_MONITOR_REPEAT_MINUTES:=5}"
    : "${HANG_MONITOR_CORE_EXEC_CMD:=}"

    echo "hang_monitor: start (stall=${HANG_MONITOR_STALL_MINUTES}m, \
pre-walltime=${HANG_MONITOR_TIMEOUT}m, logs=${HANG_MONITOR_LOG_GLOB})"

    sleep 60  # let the job actually start

    local stall_limit=$(( HANG_MONITOR_STALL_MINUTES * 60 ))
    local last_size=-1 unchanged=0 reason="" timeleft seconds_left size

    while true; do
        timeleft="$(squeue -j "${SLURM_JOBID}" --noheader --format=%L 2>/dev/null | tr -d ' ')"
        if [ -z "$timeleft" ]; then
            echo "hang_monitor: job ${SLURM_JOBID} no longer queued; exiting without dumping"
            return 0
        fi

        seconds_left="$(timeleft_to_seconds "$timeleft")"
        if [ -n "$seconds_left" ] && [ "$seconds_left" -le $(( 60 * HANG_MONITOR_TIMEOUT )) ]; then
            reason="approaching time limit (${timeleft} left)"
            break
        fi

        if [ "$stall_limit" -gt 0 ]; then
            size="$(_hang_monitor_log_size "${HANG_MONITOR_LOG_GLOB}")"
            if [ "$size" = "$last_size" ] && [ "$size" != "0" ]; then
                unchanged=$(( unchanged + HANG_MONITOR_POLL_SECONDS ))
            else
                unchanged=0
                last_size="$size"
            fi
            if [ "$unchanged" -ge "$stall_limit" ]; then
                reason="log stalled for ${unchanged}s"
                break
            fi
        fi

        sleep "${HANG_MONITOR_POLL_SECONDS}"
    done

    echo "hang_monitor: triggering artifact collection -- ${reason}"

                                                                                   
                                                                                 
                                                                                 
                                                                                  
                                         
    local base_id="${TRACEBACKS_ID:-$(date +%Y%m%d_%H%M%S)}" pass=1
    while true; do
        export TRACEBACKS_ID="${base_id}_pass$(printf '%02d' "$pass")"
        echo "hang_monitor: pass ${pass}/${HANG_MONITOR_MAX_DUMPS} -> ${TRACEBACKS_ID}"
        bash -c "${HANG_MONITOR_EXEC_CMD}" || echo "hang_monitor: exec cmd returned $?"

        [ "$pass" -ge "$HANG_MONITOR_MAX_DUMPS" ] && break
        sleep $(( HANG_MONITOR_REPEAT_MINUTES * 60 ))

                                                                                   
                                
        if ! squeue -j "${SLURM_JOBID}" --noheader --format=%L >/dev/null 2>&1; then
            echo "hang_monitor: job gone; stopping after pass ${pass}"
            break
        fi
        size="$(_hang_monitor_log_size "${HANG_MONITOR_LOG_GLOB}")"
        if [ "$size" != "$last_size" ]; then
            echo "hang_monitor: log resumed growing; stopping after pass ${pass}"
            break
        fi
        pass=$(( pass + 1 ))
    done

                                                                                    
                                                          
    if [ -n "${HANG_MONITOR_CORE_EXEC_CMD}" ]; then
        echo "hang_monitor: collecting GPU core dumps"
        bash -c "${HANG_MONITOR_CORE_EXEC_CMD}" || echo "hang_monitor: core cmd returned $?"
    fi

    echo "hang_monitor: collection done"
}
