This repo is a collection of many of the functions, classes, and scripts that I have made

# commands relating to cluster job submissions

## bsub
`bsub` is the command used most often in relation to job submissions on the cluster
There are two ways in which it can be used from the CLI, also known as the terminal of the head node. 

The first specifies all of its options in one line. For example:

`bsub -Is bash`

This command requests an interactive job session and starts bash, which also the cluster's default command shell. All other options 
for the job submission will be set to their defaults, meaning that this session should only last 30 minutes long by default with a single cpu core.

The second way of using the `bsub` command is to create a .bat file which specifies your options for a job. (Ex: `sample_job.bat`)
To use the bat file to submit a job to the cluster, use the following:

`bsub < './sample_job.bat'`
