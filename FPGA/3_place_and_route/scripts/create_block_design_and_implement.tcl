# ZCU102 BRING-UP BLOCK DESIGN for the YOLO26s per-conv kernel.
#
# WHY THIS EXISTS AND WHY IT IS FIRST: a ZCU102 bitstream needs Vivado implementation on xczu9eg,
# and MEASURED 2026-08-24 the Basic licence cannot even SEE that part (get_parts -> NOT VISIBLE).
# So after 2026-10-16 no deployable bitstream can ever be produced here. The bitstream is therefore
# the deadline-critical artefact, ahead of every optimisation step.
#
# The kernel is runtime-parameterised per conv (it reads a manifest), so ONE bitstream runs the
# whole network - this is a bounded, one-time hardware task, not a per-experiment cost.
#
# args: <phase: validate|build>  <clock MHz>  <outdir>
set phase  [lindex $argv 0]
set fmhz   [lindex $argv 1]
set outdir [lindex $argv 2]
set part   xczu9eg-ffvb1156-2-e
set board  xilinx.com:zcu102:part0:3.4
# 4th arg = IP repo; default is the retired sol_ALL. Shipping: C:/hls_y26/prj_sol_YA128/sol_YA128/impl/ip
set iprepo [expr {$argc > 3 ? [lindex $argv 3] : "C:/hls_y26/prj_sol_ALL/sol_ALL/impl/ip"}]
# 5th arg `coh` (HW lever A, 2026-09-21): gmem_act -> S_AXI_HPC0_FPD, gmem_out -> S_AXI_HPC1_FPD, both I/O-coherent
# through the CCI (AxCACHE 1111 = write-back allocate, AxPROT 010 = non-secure, as Linux memory is), so the host
# drops its per-conv X/Y cache maintenance (Y26_COHERENT=1). Weights/scales stay on non-coherent HP1/HP3.
set coh [expr {$argc > 4 && [lindex $argv 4] eq "coh"}]

puts "### BRINGUP phase=$phase clk=${fmhz}MHz out=$outdir"
file mkdir $outdir
create_project bringup $outdir/prj -part $part -force
set_property board_part $board [current_project]
set_property ip_repo_paths $iprepo [current_project]
update_ip_catalog -rebuild

create_bd_design bd

# ---- PS, with the ZCU102 board preset (DDR4 / MIO / clocking all come from the board files) ----
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e zynq_ultra_ps_e_0]
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1"} $ps

# Four HP slave ports, one per kernel master - deliberately NO shared crossbar. This design is
# ROUTING-limited (congestion 6 at 45% LUT), so an arbitrated interconnect is exactly the wrong
# thing to add. Widths are matched to the kernel's own port widths, read out of y26_conv_top.v:
#   YA128: gmem_act 128 -> HP0 | gmem_wt 64 -> HP1 | gmem_out 256 -> HP2 (HP max is 128; the
#   SmartConnect down-sizes) | gmem_scale 32 -> HP3.  (sol_ALL was act 64 / out 128.)
# (PS naming: S_AXI_GP2=HP0, GP3=HP1, GP4=HP2, GP5=HP3; M_AXI_GP2=HPM0_LPD.)
set_property -dict [list \
  CONFIG.PSU__FPGA_PL0_ENABLE {1} \
  CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ $fmhz \
  CONFIG.PSU__USE__M_AXI_GP0 {0} \
  CONFIG.PSU__USE__M_AXI_GP1 {0} \
  CONFIG.PSU__USE__M_AXI_GP2 {1} \
  CONFIG.PSU__USE__S_AXI_GP2 {1} \
  CONFIG.PSU__USE__S_AXI_GP3 {1} \
  CONFIG.PSU__USE__S_AXI_GP4 {1} \
  CONFIG.PSU__USE__S_AXI_GP5 {1} \
  CONFIG.PSU__SAXIGP2__DATA_WIDTH {128} \
  CONFIG.PSU__SAXIGP3__DATA_WIDTH {64} \
  CONFIG.PSU__SAXIGP4__DATA_WIDTH {128} \
  CONFIG.PSU__SAXIGP5__DATA_WIDTH {32} \
] $ps
if {$coh} {
    set_property -dict [list \
      CONFIG.PSU__USE__S_AXI_GP2 {0} CONFIG.PSU__USE__S_AXI_GP4 {0} \
      CONFIG.PSU__USE__S_AXI_GP0 {1} CONFIG.PSU__USE__S_AXI_GP1 {1} \
      CONFIG.PSU__AFI0_COHERENCY {1} CONFIG.PSU__AFI1_COHERENCY {1} \
      CONFIG.PSU__SAXIGP0__DATA_WIDTH {128} CONFIG.PSU__SAXIGP1__DATA_WIDTH {128} \
    ] $ps
    puts "### coh: PS knobs mentioning coherency/snoop/CCI:"
    foreach k [list_property $ps] { if {[regexp -nocase {COHER|SNOOP|CCI|AFI_FS|SAXIGP0|SAXIGP1} $k]} { puts "###   $k = [get_property $k $ps]" } }
}

# ---- the kernel ----
create_bd_cell -type ip -vlnv xilinx.com:hls:y26_conv_top:1.0 y26_conv_top_0
if {$coh} {
    set_property -dict [list \
      CONFIG.C_M_AXI_GMEM_ACT_CACHE_VALUE {"1111"} CONFIG.C_M_AXI_GMEM_ACT_PROT_VALUE {"010"} \
      CONFIG.C_M_AXI_GMEM_OUT_CACHE_VALUE {"1111"} CONFIG.C_M_AXI_GMEM_OUT_PROT_VALUE {"010"} \
    ] [get_bd_cells y26_conv_top_0]
    foreach k {ACT_CACHE ACT_PROT OUT_CACHE OUT_PROT} { puts "### coh: $k = [get_property CONFIG.C_M_AXI_GMEM_${k}_VALUE [get_bd_cells y26_conv_top_0]]" }
}

# ---- connections. Each automation is caught so a failure names ITSELF rather than dying anonymous.
proc auto {desc args} {
    puts "### connect: $desc"
    if {[catch {apply_bd_automation {*}$args} e]} { puts "### !!! FAILED ($desc): $e" ; return 0 }
    return 1
}
auto "PS M_AXI_HPM0_LPD -> kernel s_axi_control" \
  -rule xilinx.com:bd_rule:axi4 -config [list Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} \
    Master {/zynq_ultra_ps_e_0/M_AXI_HPM0_LPD} Slave {/y26_conv_top_0/s_axi_control} \
    ddr_seg {Auto} intc_ip {New AXI Interconnect} master_apm {0}] \
  [get_bd_intf_pins y26_conv_top_0/s_axi_control]

set ports {m_axi_gmem_act S_AXI_HP0_FPD m_axi_gmem_wt S_AXI_HP1_FPD m_axi_gmem_out S_AXI_HP2_FPD m_axi_gmem_scale S_AXI_HP3_FPD}
if {$coh} { set ports {m_axi_gmem_act S_AXI_HPC0_FPD m_axi_gmem_wt S_AXI_HP1_FPD m_axi_gmem_out S_AXI_HPC1_FPD m_axi_gmem_scale S_AXI_HP3_FPD} }
foreach {m hp} $ports {
    auto "kernel $m -> PS $hp" \
      -rule xilinx.com:bd_rule:axi4 -config [list Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} \
        Master "/y26_conv_top_0/$m" Slave "/zynq_ultra_ps_e_0/$hp" \
        ddr_seg {Auto} intc_ip {New AXI SmartConnect} master_apm {0}] \
      [get_bd_intf_pins zynq_ultra_ps_e_0/$hp]
}

assign_bd_address
puts "### ---- address map ----"
foreach s [get_bd_addr_segs -quiet -of_objects [get_bd_cells y26_conv_top_0]] { puts "###   $s" }
regenerate_bd_layout
save_bd_design
puts "### ---- validate ----"
if {[catch {validate_bd_design -force} e]} { puts "### !!! VALIDATE FAILED: $e" ; exit 1 }
puts "### BD VALIDATED OK"

if {$phase eq "validate"} { puts "### validate-only phase complete, stopping before synthesis." ; exit 0 }

# ---- build ----
# SPLIT INTO prep AND run ON PURPOSE. The first attempt died on a property name
# (STEPS.WRITE_BITSTREAM.IS_ENABLED does not exist in 2026.1) AFTER the validate phase had already
# exited - i.e. validate did not cover the build-only tail. Now everything cheap and typo-prone
# happens in `prep` (about a minute), and `run` does nothing but launch. A naming mistake can no
# longer surface hours into implementation.
if {$phase eq "prep" || $phase eq "build"} {
    set bdf [get_files -quiet *bd.bd]
    if {$bdf eq ""} { puts "### !!! no bd.bd in fileset" ; exit 1 }
    make_wrapper -files $bdf -top -import
    # the generated wrapper path moved between releases, so FIND it rather than assume it
    set w [get_files -quiet *_wrapper.v]
    if {$w eq ""} { set w [get_files -quiet *_wrapper.vhd] }
    if {$w eq ""} { puts "### !!! wrapper not found after make_wrapper" ; exit 1 }
    set top [file rootname [file tail $w]]
    set_property top $top [current_fileset]
    update_compile_order -fileset sources_1
    set_property strategy Performance_NetDelay_high [get_runs impl_1]
    puts "### PREP OK: top=$top  strategy=[get_property strategy [get_runs impl_1]]"
    if {$phase eq "prep"} { puts "### prep-only, stopping before implementation." ; exit 0 }
}

launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1
puts "### impl_1 status: [get_property STATUS [get_runs impl_1]]  progress [get_property PROGRESS [get_runs impl_1]]"
open_run impl_1
report_utilization    -file $outdir/util.rpt
report_timing_summary -file $outdir/timing.rpt
report_power          -file $outdir/power.rpt
set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
puts "### SYSTEM WNS: $wns  (clock ${fmhz} MHz)"
set bit [glob -nocomplain $outdir/prj/*.runs/impl_1/*.bit]
puts "### BITSTREAM: [expr {$bit eq "" ? "NOT WRITTEN" : $bit}]"
catch {write_hw_platform -fixed -include_bit -force $outdir/y26_zcu102.xsa}
puts "### BRINGUP BUILD DONE"
