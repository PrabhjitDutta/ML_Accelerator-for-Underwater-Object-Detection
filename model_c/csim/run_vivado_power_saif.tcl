# SAIF-DRIVEN post-route power for the per-conv kernel on XCZU9EG.  (§9 item 2)
#
# Separate from run_vivado_power.tcl because the hard part here is NOT reading the SAIF - it is
# making the SAIF's instance names line up with the checkpoint's. Those two hierarchies are
# different objects and nothing warns you when they fail to match:
#
#   SAIF  (from cosim's xsim testbench):   apatb_y26_conv_top_top / AESL_inst_y26_conv_top / ...
#   DCP   (from export_design -flow impl): bd_0_wrapper / bd_0_i / <hls_ip> / inst / ...
#
# So the testbench prefix must be STRIPPED off the SAIF names and the checkpoint's own prefix must be
# supplied by setting current_instance to the DUT cell. Get either wrong and read_saif reports
# success, annotates ~nothing, and report_power silently falls back to default toggle rates for
# every unmatched net - i.e. it produces the VECTORLESS number wearing a SAIF label. That is the
# specific failure this script is built to make impossible to miss.
#
#   vivado -mode batch -source Y:/hls/csim/run_vivado_power_saif.tcl -tclargs <soldir-with-impl> <saif> [strip_path]
#
# NOTE this deliberately does NOT re-run place & route. It opens the ALREADY-ROUTED checkpoint, so
# placement, routing, utilization and timing cannot move; only the power estimate changes. The
# 39,963 LUT / 3.957 ns / 12.8 FPS result of §1y.20 stands exactly as measured.
#
# Space-free paths only - see run_hls_zu9eg.tcl's header.

if {$argc < 2} {
    puts "### ERROR: usage: ... -tclargs <solution-dir> <saif> \[strip_path\]"
    exit 1
}
set soldir [lindex $argv 0]
set saif   [lindex $argv 1]
set strip  ""
if {$argc > 2} { set strip [lindex $argv 2] }

if {![file exists $saif]} {
    puts "### ERROR: SAIF '$saif' does not exist. Refusing to fall through to a vectorless number."
    exit 1
}

set dcps [glob -nocomplain "$soldir/impl/verilog/project.runs/impl_1/*_routed.dcp"]
if {[llength $dcps] == 0} { set dcps [glob -nocomplain "$soldir/impl/**/*_routed.dcp"] }
if {[llength $dcps] == 0} {
    puts "### ERROR: no *_routed.dcp under '$soldir/impl'. Nothing to annotate."
    exit 1
}
set dcp [lindex $dcps 0]
puts "### opening ROUTED checkpoint (read-only use; no re-implementation): $dcp"
open_checkpoint $dcp

# ---- locate the HLS DUT inside the block-design wrapper -------------------------------------
# The SAIF's toggles are recorded relative to the DUT, so read_saif has to be issued with
# current_instance set to the matching cell. Search by ref_name rather than assuming the bd_0_i/...
# path, which changes with the IP name and the Vitis release.
set dut ""
foreach c [get_cells -hier -quiet] {
    if {[string match "*y26_conv_top*" [get_property REF_NAME $c]]} { set dut $c ; break }
}
if {$dut eq ""} {
    puts "### WARNING: no cell with REF_NAME matching y26_conv_top; annotating at the TOP scope."
    puts "###          Expect a low match rate. Hierarchy sample follows:"
    foreach c [lrange [get_cells -hier -quiet] 0 40] { puts "###   $c  ([get_property REF_NAME $c])" }
} else {
    puts "### DUT cell: $dut"
    current_instance $dut
}

# ---- annotate ------------------------------------------------------------------------------
puts "### read_saif $saif  (strip_path: '$strip')"
if {$strip ne ""} {
    read_saif -strip_path $strip $saif
} else {
    read_saif $saif
}
current_instance -quiet

# ---- the honesty check ---------------------------------------------------------------------
# CORRECTED 2026-08-19 after this script's first run. It used to say "check the Confidence Level".
# THAT IS THE WRONG FIELD and it misleads in exactly the direction that matters: the first run
# annotated 416 of 110,416 nets - 0% - and Confidence Level still came back **Medium**, with a
# plausible-looking 2.818 W. Confidence reflects whether Vivado had a clock constraint and an
# activity file at all, NOT whether the activity file matched anything.
#
# **The field that decides it is `Design Nets Matched`.** Parse it and say so in the log, because a
# 0%-matched run is a VECTORLESS number wearing a SAIF label - the precise failure this script was
# written to make impossible, which it then committed on its first outing.
set outdir "$soldir/impl/report/verilog"
file mkdir $outdir
report_power -file "$outdir/y26_post_route_power_saif.rpt"
report_switching_activity -file "$outdir/y26_switching_activity.rpt" -quiet

set matched "UNKNOWN"
set fh [open "$outdir/y26_post_route_power_saif.rpt" r]
while {[gets $fh line] >= 0} {
    if {[string match "*Design Nets Matched*" $line]} { set matched [string trim $line "| "] ; break }
}
close $fh
puts "### DESIGN NETS MATCHED: $matched"
if {[regexp {(\d+)%} $matched -> pct] && $pct < 20} {
    puts "### ============================================================="
    puts "### THIS IS NOT A SAIF-DRIVEN NUMBER. Only $pct% of nets were annotated."
    puts "### An RTL SAIF cannot name-match a FLATTENED implemented netlist: synthesis dissolves"
    puts "###   the HLS submodule boundaries, so grp_xxx/net becomes grp_xxx_net and only the DUT"
    puts "###   BOUNDARY matches. Vivado then PROPAGATES from those boundary nets, which is better"
    puts "###   than pure default toggle rates but is not a measurement."
    puts "### Quote it as 'boundary-annotated, internally propagated', never as SAIF-driven."
    puts "### ============================================================="
}
puts "### ---------------------------------------------------------------"
puts "### power basis: SAIF-driven, stimulus = ONE conv (22.cv2.conv, oc=256 ic=663) at 8x8"
puts "###   This is a near-worst-case DYNAMIC point, not a frame average: staging- and MAC-heavy,"
puts "###   little epilogue relative to a real frame, and it never idles on DRAM."
puts "###   QUOTE IT AS AN UPPER BOUND ON DYNAMIC POWER, NOT AS 'the' POWER."
puts "### CHECK BEFORE QUOTING: the Confidence Level line in y26_post_route_power_saif.rpt."
puts "###   Anything below High/Medium means the SAIF did not annotate and the number is vectorless."
puts "### ---------------------------------------------------------------"
exit
