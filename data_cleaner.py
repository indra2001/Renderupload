# data_cleaner.py
import re
import tempfile
from dateutil import parser as date_parser
from typing import Dict, List, Tuple
import polars as pl

SAMPLE_ROWS = 1000  # how many rows to sample for type inference
NUMERIC_THRESHOLD = 0.9  # if >=90% sample values parse as numeric -> cast
DATE_THRESHOLD = 0.75    # if >=75% sample values parse as dates -> parse as date

def sanitize_column_name(name: str) -> str:
    """Lowercase, replace spaces and special chars with underscore, collapse underscores."""
    name = name.strip()
    name = re.sub(r"[^\w]+", "_", name)      # replace non-word chars with _
    name = re.sub(r"__+", "_", name)         # collapse duplicate underscores
    name = name.strip("_").lower()
    if name == "":
        name = "col"
    return name

def infer_column_types_from_sample(df_sample: pl.DataFrame) -> Dict[str, str]:
    """
    Analyze a polars DataFrame sample to decide best-effort column types:
    returns mapping: column_name -> one of {'int','float','date','string'}
    """
    result = {}
    for col in df_sample.columns:
        ser = df_sample[col].to_list()
        non_null = [v for v in ser if v is not None and str(v).strip() != ""]
        if not non_null:
            result[col] = "string"
            continue

        # Numeric checks
        int_ok = 0
        float_ok = 0
        date_ok = 0
        total = len(non_null)

        for v in non_null[:SAMPLE_ROWS]:
            s = str(v).strip()
            # try int
            try:
                if "." not in s and (s.lstrip("+-").isdigit()):
                    _ = int(s)
                    int_ok += 1
                    float_ok += 1
                    continue
            except Exception:
                pass
            # try float
            try:
                _ = float(s.replace(",", ""))  # accept thousand separators
                float_ok += 1
                # not counting as int
            except Exception:
                pass
            # try date
            try:
                date_parser.parse(s, fuzzy=False)
                date_ok += 1
            except Exception:
                pass

        # Decide
        if total > 0 and (int_ok / total) >= NUMERIC_THRESHOLD:
            result[col] = "int"
        elif total > 0 and (float_ok / total) >= NUMERIC_THRESHOLD:
            result[col] = "float"
        elif total > 0 and (date_ok / total) >= DATE_THRESHOLD:
            result[col] = "date"
        else:
            result[col] = "string"
    return result

def clean_csv(input_path: str, output_path: str, sample_rows: int = SAMPLE_ROWS) -> None:
    """
    Read CSV lazily, sanitize column names, apply de-dupe, best-effort type casting,
    and write cleaned CSV to output_path.
    """
    # Use lazy scan for memory efficiency on big files
    lf = pl.scan_csv(input_path, try_parse_dates=False)  # don't auto parse dates on scan

    # Collect small sample for inference
    sample_df = lf.limit(sample_rows).collect()
    # original column order/names
    orig_cols = sample_df.columns

    # build mapping old->new sanitized names
    sanitized_names = [sanitize_column_name(c) for c in orig_cols]
    rename_map = {orig: new for orig, new in zip(orig_cols, sanitized_names)}

    # apply rename on lazy frame
    lf = lf.rename(rename_map)

    # refresh sample with new names
    sample_df = sample_df.rename(rename_map)

    # infer types from sample
    inferred = infer_column_types_from_sample(sample_df)

    # build list of polars expressions to cast as needed
    exprs = []
    for col in lf.columns:
        t = inferred.get(col, "string")
        if t == "int":
            # remove thousand separators and cast
            exprs.append(
                pl.col(col)
                  .str.replace_all(",", "")   # strip commas in numbers
                  .cast(pl.Int64, strict=False)
                  .alias(col)
            )
        elif t == "float":
            exprs.append(
                pl.col(col)
                  .str.replace_all(",", "")
                  .cast(pl.Float64, strict=False)
                  .alias(col)
            )
        elif t == "date":
            # try parse ISO-like / common formats as datetime then date
            # str.strptime may require format; we use a best-effort: try parse to datetime (may keep nulls)
            exprs.append(
                pl.col(col)
                  .str.strptime(pl.Datetime, fmt=None, strict=False)
                  .alias(col)
            )
        else:
            # normalize whitespace in strings and keep as is
            exprs.append(
                pl.col(col)
                  .str.strip()
                  .alias(col)
            )

    # Apply transformations lazily
    lf = lf.with_columns(exprs)

    # drop duplicate rows (all columns)
    lf = lf.unique()

    # write out - using streaming write to avoid huge memory usage
    # collect and write to CSV in one shot (polars handles CSV writing efficiently)
    # for very very large datasets, consider writing in partitions or using parquet.
    df_out = lf.collect()
    df_out.write_csv(output_path)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python data_cleaner.py input.csv output.csv")
    else:
        clean_csv(sys.argv[1], sys.argv[2])
