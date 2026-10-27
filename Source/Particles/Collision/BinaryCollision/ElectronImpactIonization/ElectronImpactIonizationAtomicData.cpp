/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "ElectronImpactIonizationAtomicData.H"

#include "Particles/Collision/BinaryCollision/BinaryCollisionUtils.H"
#include "Utils/Physics/BindingEnergiesTable.H"
#include "Utils/Physics/IonizationEnergiesTable.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"

#include <AMReX_GpuContainers.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Vector.H>

#include <array>
#include <string>

ElectronImpactIonizationAtomicData::ElectronImpactIonizationAtomicData (
    std::string const& collision_name, MultiParticleContainer const * const mypc)
{
    static_assert(utils::physics::binding_energies_tab_length
                  == utils::physics::energies_tab_length,
                  "The tables of binding energies and ionization energies must have the same layout");

    // The second species is the target ion species, whose element is given by physical_element
    auto const species_names =
        BinaryCollisionUtils::get_electron_impact_ionization_species(collision_name, mypc);
    std::string physical_element;
    const amrex::ParmParse pp_ion_species_name(species_names[1]);
    pp_ion_species_name.get("physical_element", physical_element);

    auto model = IonizationCrossSectionModel::Default;
    const amrex::ParmParse pp_collision_name(collision_name);
    pp_collision_name.query_enum_sloppy("cross_section_model", model, "-_");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        utils::physics::ion_map_ids.count(physical_element) == 1,
        "Electron-impact ionization: unknown physical element " + physical_element);
    int const element_id = utils::physics::ion_map_ids.at(physical_element);
    int const atomic_number = utils::physics::ion_atomic_numbers[element_id];
    int const offset = utils::physics::ion_energy_offsets[element_id];
    auto const* const ionization_energy = utils::physics::table_ionization_energies + offset;
    auto const* const neutral_binding_energy =
        utils::physics::table_neutral_binding_energies + offset;

    // The MBELL model is defined for the subshells 1s, 2s, 2p, 3s and 3p
    constexpr int max_mbell_atomic_number = 18;
    constexpr std::array<int, ElectronImpactIonization::n_mbell_subshells> mbell_capacity =
        {2, 2, 6, 2, 6};
    if (model == IonizationCrossSectionModel::Auto) {
        model = (atomic_number <= max_mbell_atomic_number) ?
            IonizationCrossSectionModel::MBELL : IonizationCrossSectionModel::Lotz;
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        model != IonizationCrossSectionModel::MBELL || atomic_number <= max_mbell_atomic_number,
        "Electron-impact ionization: the MBELL model is only available for atomic numbers up to "
        + std::to_string(max_mbell_atomic_number));

    amrex::Vector<int> h_subshell_offsets(atomic_number + 1);
    amrex::Vector<amrex::ParticleReal> h_occupation;
    amrex::Vector<amrex::ParticleReal> h_binding_energy;
    amrex::Vector<int> h_mbell_subshell;
    amrex::Vector<int> h_mbell_q_eff;
    for (int charge_state = 0; charge_state < atomic_number; ++charge_state) {
        h_subshell_offsets[charge_state] = static_cast<int>(h_occupation.size());

        // The ion keeps the n_electrons most bound electrons of the neutral atom.
        // The electrons with the same binding energy form a subshell.
        int const n_electrons = atomic_number - charge_state;
        amrex::Vector<int> occupation;
        amrex::Vector<double> binding_energy;
        for (int e = 0; e < n_electrons; ++e) {
            auto const energy = static_cast<double>(neutral_binding_energy[e]);
            if (e > 0 && energy == binding_energy.back()) {
                occupation.back() += 1;
            } else {
                occupation.push_back(1);
                binding_energy.push_back(energy);
            }
        }
        auto const n_subshells = static_cast<int>(occupation.size());
        if (atomic_number <= max_mbell_atomic_number) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                n_subshells <= ElectronImpactIonization::n_mbell_subshells,
                "Electron-impact ionization: unexpected number of subshells");
            for (int k = 0; k < n_subshells; ++k) {
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(occupation[k] <= mbell_capacity[k],
                    "Electron-impact ionization: unexpected occupation of a subshell");
            }
        }

        // Shift of the binding energies, such that the binding energy of the outermost
        // electron is the ionization energy of the ion (Carlson et al., 1970)
        double const shift = static_cast<double>(ionization_energy[charge_state])
                             - binding_energy.back();

        // The subshells are stored from the least bound to the most bound one, so that the
        // binding energies are in increasing order
        int n_electrons_up_to_subshell = n_electrons;
        for (int k = n_subshells - 1; k >= 0; --k) {
            h_occupation.push_back(static_cast<amrex::ParticleReal>(occupation[k]));
            h_binding_energy.push_back(static_cast<amrex::ParticleReal>(binding_energy[k] + shift));
            h_mbell_subshell.push_back(atomic_number <= max_mbell_atomic_number ? k : -1);
            h_mbell_q_eff.push_back(atomic_number - n_electrons_up_to_subshell);
            n_electrons_up_to_subshell -= occupation[k];
        }
    }
    h_subshell_offsets[atomic_number] = static_cast<int>(h_occupation.size());

    // Transfer the data to the GPU device
    m_subshell_offsets.resize(h_subshell_offsets.size());
    m_occupation.resize(h_occupation.size());
    m_binding_energy.resize(h_binding_energy.size());
    m_mbell_subshell.resize(h_mbell_subshell.size());
    m_mbell_q_eff.resize(h_mbell_q_eff.size());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_subshell_offsets.begin(),
                          h_subshell_offsets.end(), m_subshell_offsets.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_occupation.begin(),
                          h_occupation.end(), m_occupation.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_binding_energy.begin(),
                          h_binding_energy.end(), m_binding_energy.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_mbell_subshell.begin(),
                          h_mbell_subshell.end(), m_mbell_subshell.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_mbell_q_eff.begin(),
                          h_mbell_q_eff.end(), m_mbell_q_eff.begin());
    amrex::Gpu::streamSynchronize();

    m_data.m_atomic_number = atomic_number;
    m_data.m_model = model;
    m_data.m_subshell_offsets = m_subshell_offsets.dataPtr();
    m_data.m_occupation = m_occupation.dataPtr();
    m_data.m_binding_energy = m_binding_energy.dataPtr();
    m_data.m_mbell_subshell = m_mbell_subshell.dataPtr();
    m_data.m_mbell_q_eff = m_mbell_q_eff.dataPtr();
}
