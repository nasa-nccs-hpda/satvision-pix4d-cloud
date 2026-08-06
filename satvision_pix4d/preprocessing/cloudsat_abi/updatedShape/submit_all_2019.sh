#!/bin/bash

# Loop through all days in a non-leap year (1 to 365)
for day in $(seq 1 365); do
  end_day=$((day + 1))
  sbatch process_day_2019.sh $day $end_day
done

echo "All 365 jobs have been submitted for 2019!"
