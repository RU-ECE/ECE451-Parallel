# Instructions to log onto Amarel

1. If you are off campus, you need to setup vpn.
  * download Cisco Anyconnect from software.rutgers.edu and install
  * I have found unfortunately that the vpn is unreliable. Hopefully that's temporary. Please report any outages to me, and to IT. But realize there isn't much we can do about it.
  * Supposedly, what I want is a "split tunnel". IT claims that it works for Windows. But clearly on my linux box, all my traffic is going to rutgers. NOT GOOD
  * connection goes up and down 5-10 times an hour. You could lose connection, so...
    * setup ssh to not timeout.
    * use termux.
    * use an editor like vscode on your computer that writes remotely to Amarel
    
1. [Instructions for using Amarel](https://sites.google.com/view/cluster-user-guide#h.3wg2loo92bhn)
1. my own quick guide
  * Log into amarel with your rutgers username and password

```bash
ssh dk9999@amarel
ssh dk9999@amarel.hpc.rutgers.edu
```

To copy your ssh-key to amarel

```bash
ssh-copy-id dk9999@amarel
```
from then on login will work without a password from your computer

The full name of the amarel cluster is amarel.hpc.rutgers.edu but the above seems to work when I am on the vpn.

  * load modules for the software you want. Note that they seem to have some ancient packages. gcc-5? So no c++20 on there for now. But cuda appears to be modern 12.1

```bash
module available        # see which modules are available to load
module load cuda        # load the latest cuda software
module load openmpi     # load openmp
```

  * To run a job, use SLURM. Here is a sample script
```bash
#!/bin/bash
# submit this job by giving the following command:
# sbatch hello_job.slurm
#SBATCH --job-name=test1_job
#SBATCH --output=test1.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00

# Load any necessary modules (if required)
# module load openmpi

# Run the compiled program
srun  --mpi=openmpi  ./test1
```

  * To run
```bash
sbatch test1_job.slurm
```
