# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_1d_electron_impact_ionization  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_electron_impact_ionization  # inputs
    "analysis.py"  # analysis
    "analysis_default_regression.py --path diags/diag1001000"  # checksum
    OFF  # dependency
)
