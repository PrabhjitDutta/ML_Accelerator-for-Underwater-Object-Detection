# COSIM SWEEP: measure per-conv cycles at REAL geometry, without re-synthesising per conv.
#
# WHY THIS EXISTS. run_hls_zu9eg.tcl does `open_solution -reset`, which wipes and re-synthesises on
# every invocation - ~18 min of csynth for a cosim that may take 2. The RTL is IDENTICAL across
# convs (the kernel is one generic engine driven by runtime config; only the tb argv changes), so
# one synthesis serves the whole sweep. This tcl opens an ALREADY-SYNTHESISED solution and calls
# cosim_design repeatedly.
#
# CRITICAL: cosim_design OVERWRITES sim/report/y26_conv_top_cosim.rpt on every call. The report is
# harvested into a per-conv file immediately after each run. Without that the sweep silently keeps
# only the LAST conv's number and every earlier result is lost with no error.
#
# NO add_files HERE, deliberately. run_hls_zu9eg.tcl:90 records that re-adding sources to an
# existing project makes llvm-link fail with "symbol multiply defined". Opening without -reset and
# without re-adding is the only safe way to reuse a synthesised solution.
#
# Y26_SWEEP_LIST : file of "<conv name> <SP> <predicted cycles>" lines, in priority order.
# Y26_SWEEP_N    : how many lines to run (0 = all). Phase 1 = 5.
# Y26_SWEEP_OUT  : directory for the harvested per-conv reports.
set sol  $::env(Y26_SOL)
set wts  "Y:/hls/weights_sq_compact_fold"
set list $::env(Y26_SWEEP_LIST)
set n    [expr {[info exists ::env(Y26_SWEEP_N)] && $::env(Y26_SWEEP_N) ne "" ? $::env(Y26_SWEEP_N) : 0}]
set outd $::env(Y26_SWEEP_OUT)
file mkdir $outd

open_project "prj_$sol"
open_solution $sol

set fh [open $list r]
set lines [split [string trim [read $fh]] "\n"]
close $fh
if {$n > 0} { set lines [lrange $lines 0 [expr {$n - 1}]] }

set rpt "prj_$sol/$sol/sim/report/y26_conv_top_cosim.rpt"
set i 0
foreach ln $lines {
    incr i
    set f [split [string trim $ln]]
    set nm [lindex $f 0]
    set sp [lindex $f 1]
    set pr [lindex $f 2]
    puts "### SWEEP \[$i/[llength $lines]\] $nm  SP=$sp  predicted $pr"
    # A failure on one conv must not abort the sweep - the remaining convs are still worth having.
    if {[catch {cosim_design -argv "$wts $sp 1 $nm" -trace_level none} err]} {
        puts "### SWEEP-FAIL $nm : $err"
        continue
    }
    if {[file exists $rpt]} {
        file copy -force $rpt "$outd/${nm}.rpt"
        puts "### SWEEP-OK $nm -> $outd/${nm}.rpt"
    } else {
        puts "### SWEEP-NORPT $nm"
    }
}
puts "### SWEEP DONE"
