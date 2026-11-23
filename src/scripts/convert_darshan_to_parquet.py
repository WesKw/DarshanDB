import argparse
import logging

import darshan
from darshan.backend import cffi_backend as darshan_backend
import pandas
import pyarrow
import pyarrow.parquet

from pathlib import Path


logger = logging.getLogger(__name__)

SUPPORTED_MODULES = {"POSIX", "STDIO", "LUSTRE", "HEATMAP"}
SUPPORTED_DXT_MODULES = {"DXT_POSIX", "DXT_MPIIO"}


def _file_paths_to_table(report: darshan.DarshanReport) -> pyarrow.Table:
    """Name records: one row per record ``id`` with its file path, to join against modules."""
    report.read_name_records()
    names = report.name_records
    return pyarrow.table({
        "id": pyarrow.array(list(names.keys()), type=pyarrow.uint64()),
        "path": pyarrow.array(list(names.values()), type=pyarrow.string()),
    })


def _module_to_table(report: darshan.DarshanReport, module: str) -> pyarrow.Table:
    """Counter based modules (POSIX, STDIO, MPI-IO, ...): one row per (record, rank)."""
    report.mod_read_all_records(module, dtype="numpy", warnings=False)
    dfs = report.records[module].to_df()
    counters = dfs["counters"]
    fcounters = dfs.get("fcounters")
    if fcounters is not None and not fcounters.empty:
        df = counters.merge(fcounters, on=["id", "rank"], how="outer")
    else:
        df = counters
    return pyarrow.Table.from_pandas(df, preserve_index=False)


def _dxt_to_table(report: darshan.DarshanReport, module: str) -> pyarrow.Table:
    """DXT modules: exploded so that every read/write operation is its own row."""
    report.mod_read_all_dxt_records(module, dtype="dict", warnings=False)
    records = report.records[module]

    columns = {
        "id": [],
        "rank": [],
        "operation": [],
        "segment": [],
        "offset": [],
        "length": [],
        "start_time": [],
        "end_time": [],
    }
    extra_segment_fields: set[str] = set()

    for i in range(len(records)):
        rec = records[i]
        for operation in ("read", "write"):
            for segment, seg in enumerate(rec[f"{operation}_segments"]):
                columns["id"].append(rec["id"])
                columns["rank"].append(rec["rank"])
                columns["operation"].append(operation)
                columns["segment"].append(segment)
                columns["offset"].append(seg["offset"])
                columns["length"].append(seg["length"])
                columns["start_time"].append(seg["start_time"])
                columns["end_time"].append(seg["end_time"])
                # Newer darshan versions can attach extra per-segment fields
                # (e.g. pthread_id / extra_info); keep them as columns too.
                for key in seg.keys() - columns.keys():
                    extra_segment_fields.add(key)

    if extra_segment_fields:
        # Rare path: second pass to collect extra fields, aligned with the rows above.
        for key in sorted(extra_segment_fields):
            columns[key] = [
                seg.get(key)
                for i in range(len(records))
                for operation in ("read", "write")
                for seg in records[i][f"{operation}_segments"]
            ]

    schema = {
        "id": pyarrow.uint64(),
        "rank": pyarrow.int64(),
        "hostname": pyarrow.string(),
        "operation": pyarrow.string(),
        "segment": pyarrow.int64(),
        "offset": pyarrow.int64(),
        "length": pyarrow.int64(),
        "start_time": pyarrow.float64(),
        "end_time": pyarrow.float64(),
    }
    return pyarrow.table(
        {name: pyarrow.array(values, type=schema.get(name)) for name, values in columns.items()}
    )


def _heatmap_to_table(report: darshan.DarshanReport) -> pyarrow.Table:
    """HEATMAP: one row per (heatmap module, rank, operation, time bin)."""
    report.read_all_heatmap_records()
    frames = []
    for heatmap_module, heatmap in report.heatmaps.items():
        bin_width = heatmap._bin_width_seconds
        for operation in ("read", "write"):
            df = heatmap.to_df(ops=[operation], interval_index=False)
            df = df.reset_index().melt(id_vars="rank", var_name="bin", value_name="bytes")
            df.insert(0, "module", str(heatmap_module))
            df.insert(2, "operation", operation)
            df["bin_start"] = df["bin"] * bin_width
            df["bin_end"] = (df["bin"] + 1) * bin_width
            frames.append(df)
    df = pandas.concat(frames, ignore_index=True)
    df = df.astype({"rank": "int64", "bin": "int64", "bytes": "int64"})
    return pyarrow.Table.from_pandas(df, preserve_index=False)


def _lustre_to_table(report: darshan.DarshanReport) -> pyarrow.Table:
    """LUSTRE: one row per file stripe component."""
    # pydarshan's dtype="pandas" combines the per-record frames itself, but their "id"
    # columns mix int64/uint64 and get promoted to float64, corrupting the ids. Read
    # the records individually and normalise "id" to uint64 before combining.
    frames = []
    while (rec := darshan_backend.log_get_record(report.log, "LUSTRE", dtype="pandas")) is not None:
        frames.append(rec["components"].astype({"id": "uint64"}))
    df = pandas.concat(frames, ignore_index=True)
    return pyarrow.Table.from_pandas(df, preserve_index=False)


def _job_to_table(report: darshan.DarshanReport) -> pyarrow.Table:
    """Job metadata: a single row with every job field as a column."""
    job = dict(report.metadata["job"])
    # Free-form key/value metadata (lib_ver, MPI-IO hints, ...) can differ between
    # logs, so it is kept as a map column to keep the schema stable.
    metadata = job.pop("metadata", {}) or {}
    row = {key: [value] for key, value in job.items()}
    row["exe"] = [report.metadata.get("exe")]

    for k, v in metadata.items():
        row[str(k)] = pyarrow.array(
            [str(v)], type=pyarrow.string()
        )

    return pyarrow.table(row)


def _write(table: pyarrow.Table, path: Path, name: str) -> None:
    pyarrow.parquet.write_table(table, path)
    logger.info("Wrote %s (%d rows) to %s", name, table.num_rows, path)


def convert_log(log: Path, output_dir: Path | None = None) -> dict[str, Path]:
    """Convert a darshan log into one parquet file per module, plus the job metadata
    and a record id -> file path mapping.

    Files are written to ``output_dir/<MODULE>.parquet``, ``output_dir/JOB.parquet`` and
    ``output_dir/FILE_PATHS.parquet`` (default: ``./<log stem>/``). Returns a mapping of
    module name (or ``"JOB"`` / ``"FILES"``) to written file.
    """
    log = Path(log)
    output_dir = Path(output_dir) if output_dir is not None else Path.cwd() / log.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    written = {}
    with darshan.DarshanReport(str(log), read_all=False) as report:
        path = output_dir / "FILES.parquet"
        _write(_file_paths_to_table(report), path, "FILES")
        written["FILES"] = path

        path = output_dir / "JOB.parquet"
        _write(_job_to_table(report), path, "JOB")
        written["JOB"] = path

        for module in report.modules:
            table = None # init

            if module in SUPPORTED_DXT_MODULES:
                table = _dxt_to_table(report, module)
            elif module in SUPPORTED_MODULES:
                if module == "HEATMAP":
                    table = _heatmap_to_table(report)
                elif module == "LUSTRE":
                    table = _lustre_to_table(report)
                else:
                    table = _module_to_table(report, module)
            else:
                logger.warning("Skipping unsupported module %s in %s", module, log)
                continue

            if not table: # we should always fail if a module passes every check but no table is created.
                raise Exception(f"Something went wrong. Could not create table for module: {module}")

            path = output_dir / f"{module}.parquet"
            _write(table, path, module)
            written[module] = path

    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__ or "Convert darshan logs to parquet files. Each module is split into its own parquet file. This is an adjustment step towards converting darshan files into a warehouse/datalakehouse-friendly format."
    )
    parser.add_argument("log_directory", type=Path, help="Root directory of all darshan log files.")
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=Path.cwd(),
        help="directory to write into; each log gets a <log stem>/ subdirectory",
    )
    parser.add_argument("-r", "--recursive", action="store_true", help="Traverse root directory recursively.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    pattern = "**/*.darshan" if args.recursive else "*.darshan"
    logs = sorted(args.log_directory.glob(pattern))
    if not logs:
        logger.warning("No darshan logs found in %s", args.log_directory)

    for log in logs:
        convert_log(log, args.output_dir / log.stem)
