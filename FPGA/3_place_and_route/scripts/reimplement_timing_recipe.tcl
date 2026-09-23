# Re-implement the block design for more timing margin. Adds impl_2 beside impl_1 from the same synth_1
# (impl_1 and its outputs are untouched). Recipe: opt Explore, power_opt on, place ExtraNetDelay_high,
# phys_opt AggressiveExplore, route NoTimingRelaxation, post-route phys_opt default.
# args: <project dir> <outdir>
set prj    [lindex $argv 0]
set outdir [lindex $argv 1]
file mkdir $outdir
open_project $prj/bringup.xpr

if {[get_runs -quiet impl_2] ne ""} { puts "### !!! impl_2 already exists - refusing to overwrite it"; exit 1 }
create_run impl_2 -parent_run synth_1 -flow [get_property FLOW [get_runs impl_1]]
set r [get_runs impl_2]
# Every property is set inside catch so a name missing in this release fails in the first minute.
foreach {p v} {
    STEPS.OPT_DESIGN.ARGS.DIRECTIVE                 Explore
    STEPS.POWER_OPT_DESIGN.IS_ENABLED               true
    STEPS.PLACE_DESIGN.ARGS.DIRECTIVE               ExtraNetDelay_high
    STEPS.PHYS_OPT_DESIGN.IS_ENABLED                true
    STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE            AggressiveExplore
    STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE               NoTimingRelaxation
    STEPS.POST_ROUTE_PHYS_OPT_DESIGN.IS_ENABLED     true
    STEPS.POST_ROUTE_PHYS_OPT_DESIGN.ARGS.DIRECTIVE Default
} {
    if {[catch {set_property $p $v $r} e]} { puts "### !!! cannot set $p = $v : $e"; exit 1 }
    puts "### impl_2 $p = [get_property $p $r]"
}

launch_runs impl_2 -to_step write_bitstream -jobs 8
wait_on_run impl_2
puts "### impl_2 status: [get_property STATUS $r]  progress [get_property PROGRESS $r]"
open_run impl_2
report_utilization    -file $outdir/util.rpt
report_timing_summary -file $outdir/timing.rpt
report_power          -file $outdir/power.rpt
puts "### SYSTEM WNS: [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]"
set bit [glob -nocomplain $prj/bringup.runs/impl_2/*.bit]
puts "### BITSTREAM: [expr {$bit eq "" ? "NOT WRITTEN" : $bit}]"
catch {write_hw_platform -fixed -include_bit -force $outdir/y26_zcu102.xsa}
puts "### HARDEN DONE"
