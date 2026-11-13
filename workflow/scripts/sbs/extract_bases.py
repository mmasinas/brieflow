from lib.sbs.extract_bases import extract_bases
from lib.shared.image_io import read_image

# Validate required params
for _param_name in ["threshold_peaks", "bases"]:
    if getattr(snakemake.params, _param_name, None) is None:
        raise ValueError(f"Required config parameter '{_param_name}' is not set")

# Load peaks data
peaks_data = read_image(snakemake.input[0])

# Load max filtered data
max_filtered_data = read_image(snakemake.input[1])

# Load cells data
cells_data = read_image(snakemake.input[2])

# Optional pixel trace (sbs.keep_call_intensities): aligned and LoG-filtered images,
# inputs 3 and 4, with the extra (non-base) channels removed to match max_filtered_data
trace_data = None
if getattr(snakemake.params, "keep_pixel_trace", False):
    import numpy as np

    extra = list(snakemake.params.extra_channel_indices or [])
    trace_data = {
        "aligned": np.delete(read_image(snakemake.input[3]), extra, axis=1),
        "log_filtered": np.delete(read_image(snakemake.input[4]), extra, axis=1),
    }

# Extract bases
# Build wildcards dict, synthesizing 'well' from 'row'+'col' in zarr mode
wc = dict(snakemake.wildcards)
if "row" in wc and "col" in wc and "well" not in wc:
    wc["well"] = wc["row"] + wc["col"]

bases_data = extract_bases(
    peaks_data=peaks_data,
    max_filtered_data=max_filtered_data,
    cells_data=cells_data,
    threshold_peaks=snakemake.params.threshold_peaks,
    bases=snakemake.params.bases,
    wildcards=wc,
    trace_data=trace_data,
)

# Save bases data
bases_data.to_csv(snakemake.output[0], index=False, sep="\t")
