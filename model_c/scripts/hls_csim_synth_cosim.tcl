# C-simulation + C-synthesis (+ optional cosim/impl) of the per-conv top on the ZCU102 part (xczu9eg-ffvb1156-2-e).
#
# Setup (neither survives a reboot):
#   subst Y: "C:\Users\Prabhjit Dutta\Desktop\MTP\Project\UOD\repo"
#   $env:XILINXD_LICENSE_FILE = "<license files>"      # must be set in the calling process
#   cd C:\hls_y26 ; vitis-run --mode hls --tcl Y:/model_c/scripts/hls_csim_synth_cosim.tcl
# vitis-run is on PATH only after settings64.bat (or call <Vitis>\bin\vitis-run.bat). Under PowerShell a
# missing command leaves $LASTEXITCODE unchanged, so check for the report files, not the exit code.
#
# Sources must be on a space-free path: csim generates a makefile that does not quote source paths. A subst
# drive works (HLS records Y:\ as is); a junction does not (HLS resolves it). The project directory is outside
# the repo so build outputs never land in the tree.

# Y26_FLAGS: the -D flags to synthesize (the shipping set is FPGA/1_hls_synthesis/hls_build_flags.txt).
#   -DY26_FX_DEQUANT  fixed-point dequant (tolerance-gated by tb_s2); without it the float dequant is bit-exact
#   -DY26_SILU_LUT    fixed-point SiLU LUT (with FX_DEQUANT)
# Passed by environment variable because vitis-run rejects extra arguments.
set y26_flags ""
set y26_sol   "sol_zu9eg"
# Every env read is guarded with `ne ""` (see Y26_COSIM below).
if {[info exists ::env(Y26_FLAGS)] && $::env(Y26_FLAGS) ne ""} { set y26_flags $::env(Y26_FLAGS) }
if {[info exists ::env(Y26_SOL)]   && $::env(Y26_SOL)   ne ""} { set y26_sol   $::env(Y26_SOL)   }
puts "### synthesizing solution '$y26_sol' with flags '$y26_flags'"
# Refuse a build at header defaults: that is not the shipping design.
if {$y26_flags eq ""} {
    if {![info exists ::env(Y26_ALLOW_DEFAULT_FLAGS)] || $::env(Y26_ALLOW_DEFAULT_FLAGS) ne "1"} {
        puts "### ERROR: Y26_FLAGS is empty. This would synthesize HEADER DEFAULTS, not the"
        puts "###        shipping config.  set it to FPGA/1_hls_synthesis/hls_build_flags.txt,"
        puts "###        or set Y26_ALLOW_DEFAULT_FLAGS=1 if you really mean defaults."
        exit 1
    }
    puts "### WARNING: building HEADER DEFAULTS by explicit request"
}

# Source root on the subst drive; overridable, but must stay space-free.
set y26_src "Y:/model_c"
set y26_wts "Y:/model_c/weights"
if {[info exists ::env(Y26_SRC)] && $::env(Y26_SRC) ne ""} { set y26_src $::env(Y26_SRC) }
if {[info exists ::env(Y26_WTS)] && $::env(Y26_WTS) ne ""} { set y26_wts $::env(Y26_WTS) }

# Fail early if the sources are not found (subst not run, or a path with a space).
if {![file exists "$y26_src/hls_kernel/conv_engine.cpp"]} {
    puts "### ERROR: no sources at '$y26_src'. Run:  subst Y: \"<repo root>\""
    exit 1
}
if {[string first " " "$y26_src$y26_wts"] >= 0} {
    puts "### ERROR: source/weights path contains a space - csim's generated Makefile will break."
    exit 1
}

# One project per variant: reusing a project without -reset re-adds sources ("symbol multiply defined"),
# and -reset would delete sibling solutions.
open_project "prj_$y26_sol" -reset
add_files $y26_src/hls_kernel/conv_engine_top.cpp -cflags "-I$y26_src -std=c++14 $y26_flags"
add_files $y26_src/hls_kernel/conv_engine.cpp     -cflags "-I$y26_src -std=c++14 $y26_flags"

# The testbench: the same conv_engine_tb.cpp the g++ gates use. It calls y26_conv_top(), which cosim needs.
# Not compiled with -fopenmp (the parallel loop is host-side scoring).
add_files -tb $y26_src/testbench/conv_engine_tb.cpp -cflags "-I$y26_src -std=c++14 $y26_flags"

set_top y26_conv_top

open_solution $y26_sol -reset -flow_target vivado
set_part {xczu9eg-ffvb1156-2-e}
# Clock period in ns; 4 = 250 MHz, the design target.
set y26_period 4
if {[info exists ::env(Y26_PERIOD)] && $::env(Y26_PERIOD) ne ""} { set y26_period $::env(Y26_PERIOD) }
puts "### clock period ${y26_period} ns"
create_clock -period $y26_period -name default

# Y26_WIDEN: m_axi max widen bitwidth (default 0 = no automatic widening), e.g. $env:Y26_WIDEN = "512".
if {[info exists ::env(Y26_WIDEN)] && $::env(Y26_WIDEN) ne "0" && $::env(Y26_WIDEN) ne ""} {
    puts "### config_interface -m_axi_max_widen_bitwidth $::env(Y26_WIDEN)"
    config_interface -m_axi_max_widen_bitwidth $::env(Y26_WIDEN)
}

# argv to the tb: <weights-dir> <spatial> [max-convs]. csim is far slower than native g++ (the -fhls-csim
# instrumentation), so it runs 12 convs by default; run_kernel_gates.sh covers all 100.
# Full coverage: $env:Y26_TB_ARGV = "Y:/model_c/weights 16"
set y26_tb_argv "$y26_wts 16 12"
set y26_cosim   0
if {[info exists ::env(Y26_TB_ARGV)] && $::env(Y26_TB_ARGV) ne ""} { set y26_tb_argv $::env(Y26_TB_ARGV) }
# `ne ""` is required: clearing an env var from PowerShell can leave it EMPTY, and `if {""}` then throws
# after csynth_design. Prefer $env:Y26_COSIM = "0".
if {[info exists ::env(Y26_COSIM)]   && $::env(Y26_COSIM)   ne ""} { set y26_cosim   $::env(Y26_COSIM)   }

# -clean forces a rebuild (no stale binary). -O compiles the host side optimized; it does not change the
# ap_int/ap_fixed semantics under test.
csim_design -argv $y26_tb_argv -clean -O
csynth_design

# Co-simulation is opt-in ($env:Y26_COSIM = 1); use a small argv, e.g. $env:Y26_TB_ARGV = "Y:/model_c/weights 8 1".
# Cosim's AXI slave has zero memory latency by default. $env:Y26_USER_STALL points at a JSON file (same
# schema as the generated random_stall.json) that injects read delay.
set y26_user_stall ""
if {[info exists ::env(Y26_USER_STALL)] && $::env(Y26_USER_STALL) ne ""} { set y26_user_stall $::env(Y26_USER_STALL) }
if {$y26_cosim} {
    if {$y26_user_stall ne ""} {
        if {![file exists $y26_user_stall]} {
            puts "### ERROR: Y26_USER_STALL='$y26_user_stall' does not exist"
            exit 1
        }
        puts "### cosim with user_stall '$y26_user_stall'"
        cosim_design -argv $y26_tb_argv -trace_level none -user_stall $y26_user_stall
    } else {
        cosim_design -argv $y26_tb_argv -trace_level none
    }
}

# Vivado implementation, opt-in ($env:Y26_IMPL = 1): post-route utilization and timing (csynth estimates
# are unreliable in both directions) plus a vectorless report_power. vivado_power_report.tcl re-reports
# with a SAIF.
set y26_impl 0
if {[info exists ::env(Y26_IMPL)] && $::env(Y26_IMPL) ne ""} { set y26_impl $::env(Y26_IMPL) }
if {$y26_impl} {
    # Implementation strategy. -vivado_phys_opt all runs post-place and post-route phys_opt. Both are overridable;
    # compare utilization only across runs with the same strategy.
    set y26_strategy "Performance_ExplorePostRoutePhysOpt"
    set y26_physopt  "all"
    if {[info exists ::env(Y26_STRATEGY)] && $::env(Y26_STRATEGY) ne ""} { set y26_strategy $::env(Y26_STRATEGY) }
    if {[info exists ::env(Y26_PHYSOPT)]  && $::env(Y26_PHYSOPT)  ne ""} { set y26_physopt  $::env(Y26_PHYSOPT)  }
    puts "### impl strategy '$y26_strategy', phys_opt '$y26_physopt'"
    config_export -vivado_impl_strategy $y26_strategy
    config_export -vivado_phys_opt      $y26_physopt
    export_design -format ip_catalog -flow impl -rtl verilog
}
exit
