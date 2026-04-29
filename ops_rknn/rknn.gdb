# break rknn_run
# commands 
#     printf "rknn_run============\n"
#     shell python3 dump.py 1
#     shell python3 dump.py 2 | grep -v "0x00000000" 
#     shell python3 dump.py 2 | grep EMIT | sed 's/\x1B\[[0-9;]*[a-zA-Z]//g' | sed 's/^.*EMIT(/EMIT(/' | grep -v "0x00000000" > /tmp/ops_rknn_emit
#     printf "rknn_run============\n"
#     continue
# end

break rknn_destroy
commands
    printf "rknn_destroy============\n"
    shell python3 dump.py 2 
   shell python3 dump.py 2 | grep -E "\[00.*]" | tee /tmp/ops_rknn_weight
#   shell python3 dump.py 2 | grep -v -E "\[00.*]" 
   shell python3 dump.py 2 | grep EMIT | sed 's/\x1B\[[0-9;]*[a-zA-Z]//g' | sed 's/^.*EMIT(/EMIT(/' | grep -v "0x00000000" > /tmp/ops_rknn_emit

#    shell python3 dump.py 3
    shell python3 dump.py 4  | grep -E "\[00.*]" | tee /tmp/ops_rknn_input
#    shell python3 dump.py 5
#    shell python3 dump.py 6
    printf "rknn_destroy============\n"
    continue
end

run
q