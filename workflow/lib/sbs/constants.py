"""Constants relevant to SBS data processing."""

# Constants
PREFIX = "prefix"
SGRNA = "sgRNA"
GENE_SYMBOL = "gene_symbol"
GENE_ID = "gene_id"
WELL = "well"
TILE = "tile"
CELL = "cell"
READ = "read"
BARCODE = "barcode"
BARCODE_COUNT = "barcode_count"
BARCODE_0 = "cell_barcode_0"
BARCODE_1 = "cell_barcode_1"
BARCODE_COUNT_0 = "cell_barcode_count_0"
BARCODE_COUNT_1 = "cell_barcode_count_1"
UMI = "UMI"
UMI_COUNT = "UMI_count"
UMI_0 = "cell_UMI_0"
UMI_1 = "cell_UMI_1"
UMI_COUNT_0 = "cell_UMI_count_0"
UMI_COUNT_1 = "cell_UMI_count_1"
POSITION_I = "i"
POSITION_J = "j"
CHANNEL = "channel"
CYCLE = "cycle"
INTENSITY = "intensity"

# Optional per-pixel trace columns that extract_bases adds to the bases table when
# sbs.keep_call_intensities is set (value at the read pixel in each image stage).
PIXEL_TRACE_COLUMNS = {"aligned": "aligned_C", "log_filtered": "log_C"}

# Prefixes of the optional per-read call-trace columns written by call_reads when
# sbs.keep_call_intensities is set. They describe a read, not a cell, so call_cells
# drops them.
KEPT_CALL_PREFIXES = (
    "aligned_C",  # aligned (registered, unfiltered) pixel value, per cycle/channel
    "log_C",  # LoG-filtered pixel value, per cycle/channel
    "intensity_C",  # max-filtered value = what the caller receives, per cycle/channel
    "corrected_C",  # four-color: crosstalk-corrected value argmax is taken over
    "scaled_C",  # frac: intensity after dye_scale
    "total_C",  # frac: sum of scaled dyes in the cycle
    "brightref_C",  # frac: the read's brightness reference (median total)
    "relbright_C",  # frac: total / brightref
    "bright_C",  # frac: relbright >= blank_threshold (else the cycle is blank)
    "fraction_C",  # frac: dye share of the cycle total
    "on_C",  # frac: dye called on (bright and fraction >= fraction_threshold)
    "confidence_C",  # frac: per-dye margin from its threshold, 0-1
    "distance_C",  # frac: mismatches between on/off pattern and the chosen base
    "tied_C",  # frac: on/off pattern equally near two or more bases
    "frac_",  # frac: thresholds and dye scale actually used (constant per run)
)
