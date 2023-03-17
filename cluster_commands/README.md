# commands relating to cluster job submissions

## bsub
`bsub` is the command used most often in relation to job submissions on the cluster
There are two ways in which it can be used from the CLI, also known as the terminal of the head node. 

The first specifies all of its options in one line. For example:

`bsub -Is bash`

This command requests an interactive job session and starts bash, which also the cluster's default command shell. All other options 
for the job submission will be set to their defaults, meaning that this session should only last 30 minutes long by default with a single cpu core.

The second way of using the `bsub` command is to create a .bat file which specifies your options for a job. (Ex: `job_sample.bat`)
To use the bat file to submit a job to the cluster, use the following:

`bsub < './job_sample.bat'`

_`job_sample.bat` consists of a job requesting 3 hours and 30 minutes of runtime, 2 cpu cores (ptile value must match), sets the name of the job to sample.
It is also set to create a log file of the submission or error file in the event that an error is raised.
anaconda3 is the version of anaconda used for python 3.x.
Rather than using `conda activate tensorflow-2`, we generally have to use `source activate tensorflow-2` on the cluster, or random errors may occur.
This particular job also converts a jupyter notebook into a python file, but python files can still be run just the same._

## bpeek
`bpeek <JOBID>` is a command used to show the current state of the output log for a job submission. 
If you do not specify a job submission, it will default to the most recently submitted job that is still actively running.
This can be useful for monitoring job progress and identifying errors.

## bkill
`bkill` is a command used to kill a job submission. By default, it will simply return the help information. Some notable forms for using it though are:

### `bkill -u $USER`
This command will terminate all currently actively running jobs by the current user
Your username must have permission to end jobs submitted by whichever user specified in the command call.
In other words, you shouldn't have to worry about accidentally ending someone else's job submissions

### `bkill <JOBID>`
This command will terminate a specific job specified by the job id.
Useful for if you have multiple active running jobs and do not want to terminate all of them.

# How to check remaining CPU hours

```bash
module load gold
gbalance -h -u $USER

```
_Note: you should not need to change the $USER to your username, as it is an linux alias for the username of the currently. 
However, if you wish to specify the user, there shouldn't be anything stopping you from swapping it out.
You should also be able to copy it by clicking an icon in the top right corner.
After that, it can just be directly pasted into the console.
If you do it that way, it should immediately be entered without having to hit the return button.
An extra newline character was added at the end for this exact purpose._





