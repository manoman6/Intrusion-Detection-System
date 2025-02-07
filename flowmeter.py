import subprocess
import cicflowmeter as cfm

# Define the command as a list of arguments
command = ["cicflowmeter", "-i", "awdl0", "-c", "networkdata/testdata.csv"]
# cfm [-h] (-i 'en0') [-c] 'testdata.csv'

# Run the command and capture output in real time
process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# Print output line-by-line
try:
    for line in process.stdout:
        print(line, end="")  # Print each line as it arrives
except KeyboardInterrupt:
    process.kill()

# Optionally, capture and print any errors after the process completes
stdout, stderr = process.communicate()
if stderr:
    print("Error output:")
    print(stderr)

