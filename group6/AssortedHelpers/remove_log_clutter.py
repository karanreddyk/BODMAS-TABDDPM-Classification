import os

input_file = "out.log"
temp_file = "out_cleaned.log"

with open(input_file, "r") as infile, open(temp_file, "w") as outfile:
    for line in infile:
        if not line.startswith("Sample timestep"):
            outfile.write(line)

os.replace(temp_file, input_file)  # Overwrites the original safely