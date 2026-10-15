# ZCU102 block design for the per-conv kernel, then synthesis, implementation and bitstream.
# The kernel is configured per conv at run time, so one bitstream runs the whole network.
# args: <phase: validate|prep|build>  <clock MHz>  <outdir>  [ip repo]  [coh | attn <attention ip repo>]
set phase  [lindex $argv 0]
set fmhz   [lindex $argv 1]
set outdir [lindex $argv 2]
set part   xczu9eg-ffvb1156-2-e
set board  xilinx.com:zcu102:part0:3.4
# 4th arg = kernel IP repo (<hls project>/<solution>/impl/ip).
set iprepo [expr {$argc > 3 ? [lindex $argv 3] : "C:/hls_y26/prj_sol_ALL/sol_ALL/impl/ip"}]
# 5th arg `coh`: gmem_act -> S_AXI_HPC0_FPD, gmem_out -> S_AXI_HPC1_FPD, I/O-coherent through the CCI
# (AxCACHE 1111, AxPROT 010), so the host can skip X/Y cache maintenance (Y26_COHERENT=1).
set coh [expr {$argc > 4 && [lindex $argv 4] eq "coh"}]
# 5th arg `attn` + 6th arg = the attention block's IP repo: adds y26_attn_top, control as a second slave on HPM0_LPD
# (must land at 0x80010000, the host's ATTN_REG_BASE), gmem_attn 128-bit -> S_AXI_HPC0_FPD with the same port settings
# as `coh`. The host always syncs the attention buffers.
set attn [expr {$argc > 5 && [lindex $argv 4] eq "attn"}]
if {$attn} { lappend iprepo [lindex $argv 5] }
# env Y26_OPS_IP = the ops block's IP repo (needs `attn`): adds y26_ops_top, control at 0x80030000 (the host's
# OPS_REG_BASE), its two masters onto the attention block's S_AXI_HPC0_FPD SmartConnect with the same port settings.
set ops [expr {[info exists ::env(Y26_OPS_IP)] && $::env(Y26_OPS_IP) ne ""}]
if {$ops && !$attn} { puts "### !!! the ops block needs attn" ; exit 1 }
if {$ops} { lappend iprepo $::env(Y26_OPS_IP) }
# env Y26_TWO=1: a second kernel instance (control at 0x80020000) sharing the first one's PS ports.
set two [expr {[info exists ::env(Y26_TWO)] && $::env(Y26_TWO) eq "1"}]

puts "### BRINGUP phase=$phase clk=${fmhz}MHz out=$outdir"
file mkdir $outdir
create_project bringup $outdir/prj -part $part -force
set_property board_part $board [current_project]
set_property ip_repo_paths $iprepo [current_project]
update_ip_catalog -rebuild

create_bd_design bd

# PS with the ZCU102 board preset (DDR4 / MIO / clocking from the board files).
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e zynq_ultra_ps_e_0]
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1"} $ps

# Four HP slave ports, one per kernel master, with no shared interconnect (the design is routing-limited).
#   gmem_act 128 -> HP0 | gmem_wt 64 -> HP1 | gmem_out 256 -> HP2 (SmartConnect down-sizes to 128) |
#   gmem_scale 32 -> HP3
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

if {$attn} {
    set_property -dict [list CONFIG.PSU__USE__S_AXI_GP0 {1} CONFIG.PSU__AFI0_COHERENCY {1}       CONFIG.PSU__SAXIGP0__DATA_WIDTH {128}] $ps
}

# The kernel.
create_bd_cell -type ip -vlnv xilinx.com:hls:y26_conv_top:1.0 y26_conv_top_0
if {$coh} {
    set_property -dict [list \
      CONFIG.C_M_AXI_GMEM_ACT_CACHE_VALUE {"1111"} CONFIG.C_M_AXI_GMEM_ACT_PROT_VALUE {"010"} \
      CONFIG.C_M_AXI_GMEM_OUT_CACHE_VALUE {"1111"} CONFIG.C_M_AXI_GMEM_OUT_PROT_VALUE {"010"} \
    ] [get_bd_cells y26_conv_top_0]
    foreach k {ACT_CACHE ACT_PROT OUT_CACHE OUT_PROT} { puts "### coh: $k = [get_property CONFIG.C_M_AXI_GMEM_${k}_VALUE [get_bd_cells y26_conv_top_0]]" }
}

# Direct-codes kernel (-DY26_YC): its code port takes the free HPC1 port, 64-bit, not coherent.
set yc [expr {[get_bd_intf_pins -quiet y26_conv_top_0/m_axi_gmem_code] ne ""}]
puts "### direct-codes port: $yc"
if {$yc} {
    if {$coh} { puts "### !!! coh already uses HPC1" ; exit 1 }
    set_property -dict [list CONFIG.PSU__USE__S_AXI_GP1 {1} CONFIG.PSU__SAXIGP1__DATA_WIDTH {64}] $ps
}

if {$attn} {
    create_bd_cell -type ip -vlnv xilinx.com:hls:y26_attn_top:1.0 y26_attn_top_0
    set_property -dict [list CONFIG.C_M_AXI_GMEM_ATTN_CACHE_VALUE {"1111"} CONFIG.C_M_AXI_GMEM_ATTN_PROT_VALUE {"010"}]       [get_bd_cells y26_attn_top_0]
}

# Connections. Each automation is caught so a failure names itself.
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
if {$yc} { lappend ports m_axi_gmem_code S_AXI_HPC1_FPD }
foreach {m hp} $ports {
    auto "kernel $m -> PS $hp" \
      -rule xilinx.com:bd_rule:axi4 -config [list Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} \
        Master "/y26_conv_top_0/$m" Slave "/zynq_ultra_ps_e_0/$hp" \
        ddr_seg {Auto} intc_ip {New AXI SmartConnect} master_apm {0}] \
      [get_bd_intf_pins zynq_ultra_ps_e_0/$hp]
}

if {$attn} {
    auto "PS M_AXI_HPM0_LPD -> attention s_axi_control"       -rule xilinx.com:bd_rule:axi4 -config [list Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto}         Master {/zynq_ultra_ps_e_0/M_AXI_HPM0_LPD} Slave {/y26_attn_top_0/s_axi_control}         ddr_seg {Auto} intc_ip {Auto} master_apm {0}]       [get_bd_intf_pins y26_attn_top_0/s_axi_control]
    auto "attention m_axi_gmem_attn -> PS S_AXI_HPC0_FPD"       -rule xilinx.com:bd_rule:axi4 -config [list Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto}         Master {/y26_attn_top_0/m_axi_gmem_attn} Slave {/zynq_ultra_ps_e_0/S_AXI_HPC0_FPD}         ddr_seg {Auto} intc_ip {New AXI SmartConnect} master_apm {0}]       [get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HPC0_FPD]
}

if {$ops} {
    create_bd_cell -type ip -vlnv xilinx.com:hls:y26_ops_top:1.0 y26_ops_top_0
    set_property -dict [list CONFIG.C_M_AXI_GMEM_OPA_CACHE_VALUE {"1111"} CONFIG.C_M_AXI_GMEM_OPA_PROT_VALUE {"010"} \
                             CONFIG.C_M_AXI_GMEM_OPD_CACHE_VALUE {"1111"} CONFIG.C_M_AXI_GMEM_OPD_PROT_VALUE {"010"}] [get_bd_cells y26_ops_top_0]
    auto "PS M_AXI_HPM0_LPD -> ops s_axi_control" \
      -rule xilinx.com:bd_rule:axi4 -config [list Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} \
        Master {/zynq_ultra_ps_e_0/M_AXI_HPM0_LPD} Slave {/y26_ops_top_0/s_axi_control} \
        ddr_seg {Auto} intc_ip {Auto} master_apm {0}] \
      [get_bd_intf_pins y26_ops_top_0/s_axi_control]
    set net [get_bd_intf_nets -of_objects [get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HPC0_FPD]]
    set smc [get_bd_cells -of_objects [get_bd_intf_pins -of_objects $net -filter {MODE == Master}]]
    set smc [lsearch -inline -not -exact $smc [get_bd_cells zynq_ultra_ps_e_0]]
    puts "### connect: ops m_axi_gmem_opa, m_axi_gmem_opd -> $smc S01_AXI, S02_AXI -> PS S_AXI_HPC0_FPD"
    set_property CONFIG.NUM_SI 3 $smc
    connect_bd_intf_net [get_bd_intf_pins y26_ops_top_0/m_axi_gmem_opa] [get_bd_intf_pins $smc/S01_AXI]
    connect_bd_intf_net [get_bd_intf_pins y26_ops_top_0/m_axi_gmem_opd] [get_bd_intf_pins $smc/S02_AXI]
}

if {$two} {
    create_bd_cell -type ip -vlnv xilinx.com:hls:y26_conv_top:1.0 y26_conv_top_1
    auto "PS M_AXI_HPM0_LPD -> kernel 1 s_axi_control" \
      -rule xilinx.com:bd_rule:axi4 -config [list Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} \
        Master {/zynq_ultra_ps_e_0/M_AXI_HPM0_LPD} Slave {/y26_conv_top_1/s_axi_control} \
        ddr_seg {Auto} intc_ip {Auto} master_apm {0}] \
      [get_bd_intf_pins y26_conv_top_1/s_axi_control]
    foreach {m hp} $ports {
        set net [get_bd_intf_nets -of_objects [get_bd_intf_pins zynq_ultra_ps_e_0/$hp]]
        set smc [get_bd_cells -of_objects [get_bd_intf_pins -of_objects $net -filter {MODE == Master}]]
        set smc [lsearch -inline -not -exact $smc [get_bd_cells zynq_ultra_ps_e_0]]
        puts "### connect: kernel 1 $m -> $smc S01_AXI -> PS $hp"
        set_property CONFIG.NUM_SI 2 $smc
        connect_bd_intf_net [get_bd_intf_pins y26_conv_top_1/$m] [get_bd_intf_pins $smc/S01_AXI]
    }
}
assign_bd_address
if {$two} {
    set seg [get_bd_addr_segs -of_objects [get_bd_addr_spaces zynq_ultra_ps_e_0/Data] -filter {NAME =~ *conv_top_1*}]
    if {[llength $seg] != 1} { puts "### !!! kernel 1 control segments: $seg" ; exit 1 }
    if {[get_property OFFSET $seg] != 0x80020000} { set_property offset 0x80020000 $seg }
    foreach s [get_bd_addr_segs -quiet -of_objects [get_bd_cells y26_conv_top_1]] { puts "###   $s" }
}
if {$ops} {
    set seg [get_bd_addr_segs -of_objects [get_bd_addr_spaces zynq_ultra_ps_e_0/Data] -filter {NAME =~ *ops_top*}]
    if {[llength $seg] != 1} { puts "### !!! ops control segments: $seg" ; exit 1 }
    if {[get_property OFFSET $seg] != 0x80030000} { set_property offset 0x80030000 $seg }
    foreach s [get_bd_addr_segs -quiet -of_objects [get_bd_cells y26_ops_top_0]] { puts "###   $s" }
    foreach k {OPA_CACHE OPA_PROT OPD_CACHE OPD_PROT} { puts "### ops: $k = [get_property CONFIG.C_M_AXI_GMEM_${k}_VALUE [get_bd_cells y26_ops_top_0]]" }
    foreach m {m_axi_gmem_opa m_axi_gmem_opd} {
        if {[llength [get_bd_addr_segs -quiet -of_objects [get_bd_addr_spaces y26_ops_top_0/Data_$m]]] == 0} { puts "### !!! ops $m has no memory segment" ; exit 1 }
    }
}
puts "### ---- address map ----"
foreach s [get_bd_addr_segs -quiet -of_objects [get_bd_cells y26_conv_top_0]] { puts "###   $s" }
foreach s [get_bd_addr_segs -quiet -of_objects [get_bd_addr_spaces zynq_ultra_ps_e_0/Data]] {
    puts "###   PS: $s  offset [get_property OFFSET $s]  range [get_property RANGE $s]"
    if {[string match *conv_top_0* $s] && [get_property OFFSET $s] != 0x80000000} { puts "### !!! kernel control moved" ; exit 1 }
    if {[string match *conv_top_1* $s] && [get_property OFFSET $s] != 0x80020000} { puts "### !!! kernel 1 control is not at 0x80020000" ; exit 1 }
    if {[string match *attn_top* $s] && [get_property OFFSET $s] != 0x80010000} { puts "### !!! attention control is not at 0x80010000" ; exit 1 }
    if {[string match *ops_top* $s] && [get_property OFFSET $s] != 0x80030000} { puts "### !!! ops control is not at 0x80030000" ; exit 1 }
}
if {$attn} {
    foreach s [get_bd_addr_segs -quiet -of_objects [get_bd_cells y26_attn_top_0]] { puts "###   $s" }
    foreach k {CACHE PROT} { puts "### attn: $k = [get_property CONFIG.C_M_AXI_GMEM_ATTN_${k}_VALUE [get_bd_cells y26_attn_top_0]]" }
}
regenerate_bd_layout
save_bd_design
puts "### ---- validate ----"
if {[catch {validate_bd_design -force} e]} { puts "### !!! VALIDATE FAILED: $e" ; exit 1 }
puts "### BD VALIDATED OK"

if {$phase eq "validate"} { puts "### validate-only phase complete, stopping before synthesis." ; exit 0 }

# Build. Everything cheap and typo-prone runs first (phase prep stops there, about a minute), so a
# property-name mistake cannot surface hours into implementation.
if {$phase eq "prep" || $phase eq "build"} {
    set bdf [get_files -quiet *bd.bd]
    if {$bdf eq ""} { puts "### !!! no bd.bd in fileset" ; exit 1 }
    make_wrapper -files $bdf -top -import
    # the generated wrapper path varies by release, so find it
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
