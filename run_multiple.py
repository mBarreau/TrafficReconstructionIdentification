#!/usr/bin/env python3


from argparse import ArgumentParser
import subprocess
import re
import datetime
import csv


def parse_args():
    ap = ArgumentParser(
            description="Run multiple simulations and collect time and error data")

    ap.add_argument(
            "type",
            help="Type of simulation to run",
            choices=["godunov", "sumo"])

    ap.add_argument("-N", type=int, default=50,
                    help="Number of simulations to run")

    return ap.parse_args()


def main():
    args = parse_args()

    target_path = "./main_" + args.type + ".py"

    data = []

    for run in range(args.N):
        print(f"Simulation run {run}")

        result = subprocess.run(
                ["/usr/bin/env", "python3", target_path],
                stdout=subprocess.PIPE)

        if result.returncode != 0:
            raise RuntimeError(f"Simulation run {run} failed.")

        time, err = process_output(result.stdout.decode('utf-8'))
        data.append({'time': time, 'error': err})

    output_csv = "time_and_error_" + args.type + f"_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}" + ".csv"

    with open(output_csv, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['time', 'error'])
        writer.writeheader()
        writer.writerows(data)


def process_output(output):
    time, error = 0., 0.

    time_match = re.search("([0-9]+)h\s*([0-9]+)m\s*([0-9]+)s", output)

    if time_match:
        time = 3600.0 * float(time_match.group(1)) + 60.0 * float(time_match.group(2)) + float(time_match.group(3))
    else:
        raise RuntimeError("No time data found")

    error_match = re.search("Normalized L\^2 error:\s*([0-9.,]+)", output)

    if error_match:
        error = float(error_match.group(1))
    else:
        raise RuntimeError("No error data found")

    return time, error


if __name__ == "__main__":
    main()
