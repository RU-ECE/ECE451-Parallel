# Instructions to Log onto Amarel

## 1. VPN (If You Are Off Campus)

If you are **off campus**, you must first set up the Rutgers VPN.

- Download **Cisco AnyConnect** from [software.rutgers.edu](https://software.rutgers.edu) and install it.
- The VPN can be unreliable at times:
	- Connections may go up and down 5–10 times an hour.
	- Please report outages to me and to IT, but understand there may be limited immediate fixes.
- What we really want is a **“split tunnel”**:
	- IT claims this works on Windows.
	- On some Linux setups, all traffic may go through Rutgers (not ideal).
- Because the VPN can drop:
	- Configure `ssh` to **not timeout** as quickly.
	- Consider using **Termux** (on Android) or similar tools.
	- Use an editor like **VS Code** on your local machine that edits files **remotely** on Amarel (via SSH/remote
	  extension).

---

## 2. Official Amarel Instructions

Follow the official cluster user guide here:

- <https://sites.google.com/view/cluster-user-guide#h.3wg2loo92bhn>

---

## 3. Quick Start Guide (My Version)

### 3.1. Log into Amarel

Use your Rutgers NetID and password:

```bash
ssh dk9999@amarel
# or
ssh dk9999@amarel.hpc.rutgers.edu
````

To copy your SSH key to Amarel (so you don’t have to type your password every time):

```bash
ssh-copy-id dk9999@amarel
```

After this, SSH login from your computer should work without a password.

> Note: The full hostname is `amarel.hpc.rutgers.edu`, but `amarel` alone usually works once you’re on the VPN.

---

### 3.2. Load Modules

Amarel uses environment **modules** to provide compilers, CUDA, MPI, etc.

```bash
module avail         # see which modules are available to load
module load cuda     # load the latest CUDA toolkit
module load openmpi  # load MPI library
```

Notes:

* Some packages are **old** (e.g., `gcc-5`), so modern C++20 features may not be available with the default toolchain.
* CUDA appears to be relatively modern (e.g., 12.1).

---

### 3.3. Running Jobs with SLURM

You do **not** run heavy jobs directly on the login node. Instead, you submit a **batch job** using SLURM.

Example SLURM script: `test1_job.slurm`

```bash
#!/bin/bash
# Submit this job with:
#   sbatch test1_job.slurm

#SBATCH --job-name=test1_job
#SBATCH --output=test1.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00

# Load any necessary modules (if required)
# module load openmpi

# Run the compiled program
srun --mpi=openmpi ./test1
```

To submit the job:

```bash
sbatch test1_job.slurm
```

SLURM will queue the job and run it on the cluster. The output will be written to `test1.txt`.
