# Post-route POWER, UTILIZATION and TIMING for the per-conv kernel on XCZU9EG.
#
# This is the project's FIRST power measurement - nothing has ever been measured. It is also the most
# license-window-sensitive step in the whole flow: the Vivado ML Enterprise evaluation that unlocks
# ZU9EG expires 2026-10-16. The permanent Basic license keeps Vivado_Implementation but only up to
# ZU7EV, so after that date this runs on the ZU6EG proxy or not at all.
#
#   # 1. produce a routed design (this is what writes the checkpoint we open here)
#   $env:Y26_IMPL = 1
#   vitis-run --mode hls --tcl Y:/hls/csim/run_hls_zu9eg.tcl
#
#   # 2. then, pointing at that solution
#   vivado -mode batch -source Y:/hls/csim/run_vivado_power.tcl -tclargs C:/hls_y26/prj_<sol>/<sol>
#
# Space-free paths only, for the same reason as the HLS flow - see run_hls_zu9eg.tcl's header.
#
# ---------------------------------------------------------------------------------------------
# READ THIS BEFORE QUOTING THE POWER NUMBER.
#
# With no switching-activity input, report_power runs VECTORLESS: it assumes default toggle rates
# (typically 12.5% on data nets) rather than measuring what this design actually does. That is a
# rough estimate, NOT a measurement, and its error bars are wide - Vivado itself labels the
# confidence level, and vectorless on a fresh design usually comes back "Low".
#
# The honest version needs real activity: run cosim with tracing enabled, convert the resulting
# waveform to SAIF, and read it in with read_saif before report_power. That path is set up below and
# gated on the SAIF actually existing, so the script cannot silently produce a vectorless number
# while appearing to have used activity data. ALWAYS report which of the two produced the figure -
# the confidence level is printed for exactly this reason.
# ---------------------------------------------------------------------------------------------

if {$argc < 1} {
    puts "### ERROR: usage: vivado -mode batch -source run_vivado_power.tcl -tclargs <solution-dir> \[saif\]"
    exit 1
}
set soldir [lindex $argv 0]
set saif   ""
if {$argc > 1} { set saif [lindex $argv 1] }

# Locate the routed checkpoint. Its exact path has moved between releases, so search rather than
# hard-code and fail loudly if the impl flow did not actually run to completion.
set dcps [glob -nocomplain "$soldir/impl/verilog/project.runs/impl_1/*_routed.dcp"]
if {[llength $dcps] == 0} {
    set dcps [glob -nocomplain "$soldir/impl/**/*_routed.dcp"]
}
if {[llength $dcps] == 0} {
    puts "### ERROR: no *_routed.dcp under '$soldir/impl'."
    puts "###        Did export_design -flow impl run to completion? A -flow syn export stops"
    puts "###        after synthesis and never routes, which produces no checkpoint."
    exit 1
}
set dcp [lindex $dcps 0]
puts "### opening routed checkpoint: $dcp"
open_checkpoint $dcp

set outdir "$soldir/impl/report/verilog"
file mkdir $outdir

# Utilization and timing first: both are true post-route measurements and neither depends on the
# activity question above. These are the numbers that settle whether the csynth ESTIMATES were sound.
report_utilization      -file "$outdir/y26_post_route_utilization.rpt"
report_timing_summary   -file "$outdir/y26_post_route_timing.rpt"

set confidence_note "VECTORLESS (default toggle rates) - treat as an estimate, not a measurement"
if {$saif ne "" && [file exists $saif]} {
    puts "### reading switching activity: $saif"
    read_saif $saif
    set confidence_note "SAIF-driven (activity from cosim)"
} elseif {$saif ne ""} {
    # Do not silently fall through to vectorless - that is how an estimate gets quoted as a
    # measurement. A named-but-missing SAIF is a mistake worth stopping for.
    puts "### ERROR: SAIF '$saif' was specified but does not exist."
    exit 1
}
report_power -file "$outdir/y26_post_route_power.rpt"

puts "### ---------------------------------------------------------------"
puts "### power basis: $confidence_note"
puts "### reports written to $outdir"
puts "###   y26_post_route_utilization.rpt   real LUT/FF/DSP/BRAM (vs the csynth estimate)"
puts "###   y26_post_route_timing.rpt        real Fmax, routing delay included"
puts "###   y26_post_route_power.rpt         check the Confidence Level line before quoting"
puts "### ---------------------------------------------------------------"
exit
