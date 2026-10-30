/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_IMPACT_IONIZATION_ATOMIC_DATA_H_
#define WARPX_IMPACT_IONIZATION_ATOMIC_DATA_H_

#include "ImpactIonizationCrossSection.H"

#include "Particles/Collision/ScatteringProcess.H"
#include "Particles/MultiParticleContainer_fwd.H"

#include <AMReX_GpuContainers.H>
#include <AMReX_REAL.H>

#include <memory>
#include <string>

/**
 * \brief Builds and stores (on the GPU if needed) the data used to compute the ionization cross
 * sections of the target of an electron-impact or ion-impact ionization collision: the occupied
 * subshells of each charge state of the target, with their occupation and binding energy, and,
 * for ion projectiles, the parameters of the Rudd model of each subshell and the tables of cross
 * sections.
 *
 * For an atomic target (an ion species with an ionization level), the ground-state configuration
 * of a charge state is obtained by removing the least bound electrons of the neutral atom.
 * Following Carlson et al. (At. Data 2, 63 (1970)), the binding energies of the electrons of an
 * ion are approximated by the binding energies of the corresponding electrons of the neutral
 * atom, shifted such that the binding energy of the outermost electron is equal to the ionization
 * energy of the ion. For a molecular target (ion projectiles only), the binding energies and the
 * occupations of the molecular orbitals are given by the user.
 */
class ImpactIonizationAtomicData
{
public:
    /**
     * \brief Default constructor of the ImpactIonizationAtomicData class.
     */
    ImpactIonizationAtomicData () = default;

    /**
     * \brief Constructor of the ImpactIonizationAtomicData class. For an atomic target, it reads
     * the element of the target species (<target_species>.physical_element); for a molecular
     * target, it reads the binding energies and the occupations of the molecular orbitals
     * (<collision_name>.binding_energies and <collision_name>.occupations).
     * For electron projectiles, it reads the model used for the cross sections
     * (<collision_name>.cross_section_model, where the default value, auto, selects the MBELL
     * model for atomic numbers up to 18 and the Lotz model otherwise).
     * For ion projectiles, it selects the parameters of the Rudd model of each subshell (which can
     * be set by the user for the outer subshells with <collision_name>.rudd_parameters), computes
     * the tables of cross sections and, for a molecular target, reads the total cross section
     * from the file <collision_name>.cross_section, if it is specified.
     *
     * @param[in] collision_name the name of the collision
     * @param[in] mypc pointer to the MultiParticleContainer
     */
    ImpactIonizationAtomicData (std::string const& collision_name,
                                MultiParticleContainer const * mypc);

    // The pointers in m_data refer to the device vectors owned by this object: copies are
    // therefore forbidden, while moves are allowed (they do not reallocate the data)
    ImpactIonizationAtomicData (ImpactIonizationAtomicData const&) = delete;
    ImpactIonizationAtomicData& operator= (ImpactIonizationAtomicData const&) = delete;
    ImpactIonizationAtomicData (ImpactIonizationAtomicData&&) = default;
    ImpactIonizationAtomicData& operator= (ImpactIonizationAtomicData&&) = default;
    ~ImpactIonizationAtomicData () = default;

    /**
     * \brief Returns the light-weight structure used to access the atomic data in GPU kernels
     */
    [[nodiscard]] ImpactIonization::AtomicData const& data () const { return m_data; }

private:
    amrex::Gpu::DeviceVector<int> m_subshell_offsets;
    amrex::Gpu::DeviceVector<amrex::ParticleReal> m_occupation;
    amrex::Gpu::DeviceVector<amrex::ParticleReal> m_binding_energy;
    amrex::Gpu::DeviceVector<int> m_mbell_subshell;
    amrex::Gpu::DeviceVector<int> m_mbell_q_eff;
    amrex::Gpu::DeviceVector<double> m_rudd_parameters;
    amrex::Gpu::DeviceVector<double> m_total_table;
    amrex::Gpu::DeviceVector<double> m_subshell_table;
    std::unique_ptr<ScatteringProcess> m_user_cross_section;

    ImpactIonization::AtomicData m_data;
};

#endif // WARPX_IMPACT_IONIZATION_ATOMIC_DATA_H_
