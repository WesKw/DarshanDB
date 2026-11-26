import duckdb

from argparse import ArgumentParser
from pathlib import Path


def generate_darshandb(parquet_logs: Path, db_path: Path):
    ...


"""This file takes a set of darshan logs and inserts it into a physical duckdb setup."""
if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument(
        "--parquet-logs", type=str, required=True, 
        help="Path to the parquet logs to be inserted into the database. These should be generated form convert_darshan_logs_to_parquet.py"
    )
    args.add_argument(
        "--db-path", type=str, required=True,
        help="The path to save the duckdb database to. If the database already exists, it will be appended to."
    )
    args.add_argument(
        "--overwrite", action="store_true",
        help="If the database already exists, overwrite it instead of appending to it."
    )
    args.add_argument(
        "--schema", type=str, help="The schema file to use for the database.", required=True
    )
    args = args.parse_args()

    log_path = Path(args.parquet_logs)
    duckdb_path = Path(args.db_path)
    schema = Path(args.schema)


    generate_darshandb(log_path, duckdb_path)

