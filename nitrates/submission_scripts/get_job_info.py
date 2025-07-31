import os
import argparse
import logging
import subprocess


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ssh", help="Do we need to ssh in to submit?", action="store_true"
    )
    parser.add_argument("--name", type=str, help="Name of jobs to query", default=None)
    args = parser.parse_args()
    return args


def get_jobids(args):
    cmd_list = ["qselect", "-N", args.name]
    out = subprocess.Popen(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    stdout, stderr = out.communicate()

    jobids = []
    for line in stdout.split("\n"):
        try:
            jobids.append(int(line.split(".")[0]))
        except:
            pass
    return jobids


def get_env_vars(jobid):
    # cmd = 'checkjob -v -v %d | grep EnvVariables' %(jobid)
    cmd_list = [
        "/opt/moab/bin/checkjob",
        "-v",
        "-v",
        str(jobid),
    ]  # , '|', 'grep', 'EnvVariables']
    cmd_list2 = ["grep", "EnvVariables"]
    # print cmd_list
    # cmd_list = shlex.split(cmd)
    out1 = subprocess.Popen(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = subprocess.Popen(
        cmd_list2, stdin=out1.stdout, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )

    stdout, stderr = out.communicate()

    env_var_list = stdout[14:].split(",")

    env_vars = {}
    for envvar in env_var_list:
        ev = envvar.split("=")
        env_vars[ev[0].strip()] = ev[1]

    return env_vars


def get_proc_num(jobid):
    env_vars = get_env_vars(jobid)
    return env_vars["jobid"]


def main(args):
    jobids = get_jobids(args)
    proc_nums = []
    for jobid in jobids:
        proc_nums.append(int(get_proc_num(jobid)))


if __name__ == "__main__":
    # logging.basicConfig(filename='submit_jobs_log.log', level=logging.DEBUG,\
    #                 format='%(asctime)s-' '%(levelname)s- %(message)s')
    args = cli()

    main(args)
