#!/usr/bin/env bash
# Monitor all experiment tmux sessions across M0-M3 for errors.
# Prints a one-liner status every 60s; escalates known error patterns to stdout
# so the Monitor tool can catch them.
#
# Error patterns that trigger alerts:
#   - CUDA OOM
#   - Ray address already in use / ray init
#   - Traceback / Error / FAILED / assert
#   - NaN/Inf in loss or reward
#   - verl crash signatures

REPO=/newcpfs/lxh/agentic-training/proposal_rl

ERROR_RE='(CUDA out of memory|Traceback \(most recent|RuntimeError|AssertionError|address already in use|ray\.init|FAILED|OOM killer|loss: nan|loss: inf|reward: nan|reward: inf|probability tensor contains|_dl_allocate_tls_init|ActorDiedError|SIGABRT|SIGKILL)'

check_pane() {
    local host=$1 session=$2
    local raw
    if [[ "$host" == "local" ]]; then
        raw=$(tmux capture-pane -t "$session" -p -S -30 2>/dev/null)
    else
        raw=$(ssh "$host" "tmux capture-pane -t '$session' -p -S -30 2>/dev/null" 2>/dev/null)
    fi
    # Suppress alerts if the session has already finished successfully
    if echo "$raw" | grep -qE "FINISHED SUCCESSFULLY|EXITED \(check"; then
        :
    else
        local errors
        errors=$(echo "$raw" | grep -E "$ERROR_RE" | tail -5)
        if [[ -n "$errors" ]]; then
            echo "ALERT [$session @ ${host}]: $(echo "$errors" | head -3 | tr '\n' ' | ')"
        fi
    fi
    # Extract last progress line for status
    local last
    last=$(echo "$raw" | grep -E '(Training Progress|step:|Epoch |reward|loss:|SFT|RL|CoT|hparam|Best|FINISHED|EXITED|ERROR|WARNING)' | tail -1)
    echo "STATUS [$session @ ${host}]: ${last:-<no output>}"
}

echo "=== Experiment Monitor $(date '+%Y-%m-%d %H:%M:%S') ==="
check_pane local   exp10
check_pane local   exp14
check_pane lxh_agent_0 exp09
check_pane lxh_agent_0 exp13
check_pane lxh_agent_2 exp11
check_pane lxh_agent_2 exp16
check_pane lxh_agent_3 exp12
check_pane lxh_agent_3 exp17
echo "=== End ==="
