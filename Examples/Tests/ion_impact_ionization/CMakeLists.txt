# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_1d_ion_impact_ionization  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_ion_impact_ionization  # inputs
    "analysis.py"  # analysis
    "analysis_default_regression.py --path diags/diag1000020"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_ion_impact_ionization_molecular  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_ion_impact_ionization_molecular  # inputs
    "analysis.py"  # analysis
    "analysis_default_regression.py --path diags/diag1000050"  # checksum
    OFF  # dependency
)
