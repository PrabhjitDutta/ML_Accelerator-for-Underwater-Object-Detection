# Post-route power, utilization and timing for the per-conv kernel on XCZU9EG.
#   # 1. a routed design:  $env:Y26_IMPL = 1; vitis-run --mode hls --tcl Y:/model_c/scripts/hls_csim_synth_cosim.tcl
#   # 2. vivado -mode batch -source Y:/model_c/scripts/vivado_power_report.tcl -tclargs <work dir>/prj_<sol>/<sol>
# Space-free paths only (see hls_csim_synth_cosim.tcl).
# Without a SAIF, report_power is vectorless (default toggle rates), an estimate. With a SAIF from cosim it uses
# real activity. The printed confidence level says which.

if {$argc < 1} {
    puts "### ERROR: usage: vivado -mode batch -source vivado_power_report.tcl -tclargs <solution-dir> \[saif\]"
    exit 1
}
set soldir [lindex $argv 0]
set saif   ""
if {$argc > 1} { set saif [lindex $argv 1] }

# Search for the routed checkpoint (its path varies by release).
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

# Utilization and timing do not depend on switching activity.
report_utilization      -file "$outdir/y26_post_route_utilization.rpt"
report_timing_summary   -file "$outdir/y26_post_route_timing.rpt"

set confidence_note "VECTORLESS (default toggle rates) - treat as an estimate, not a measurement"
if {$saif ne "" && [file exists $saif]} {
    puts "### reading switching activity: $saif"
    read_saif $saif
    set confidence_note "SAIF-driven (activity from cosim)"
} elseif {$saif ne ""} {
    # A named SAIF that is missing is an error, not a silent fall-back to vectorless.
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
