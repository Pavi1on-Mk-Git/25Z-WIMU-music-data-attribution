echo "GPU hours used: $(hpc-jobs-history -A $ATHENA_GRANT -d 7 | awk '$11 ~ /^[0-9.]*$/ {sum += $11} END {print sum}')"
