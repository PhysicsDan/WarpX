/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_ELECTRON_IMPACT_IONIZATION_ATOMIC_DATA_H_
#define WARPX_ELECTRON_IMPACT_IONIZATION_ATOMIC_DATA_H_

#include "ElectronImpactIonizationCrossSection.H"

#include "Particles/MultiParticleContainer_fwd.H"

#include <AMReX_GpuContainers.H>
#include <AMReX_REAL.H>

#include <string>

/**
 * \brief Builds and stores (on the GPU if needed) the atomic data used to compute the
 * electron-impact ionization cross sections of the ions of a given element: the occupied
 * subshells of each charge state, with their occupation and binding energy.
 *
 * The ground-state configuration of a charge state is obtained by removing the least bound
 * electrons of the neutral atom. Following Carlson et al. (At. Data 2, 63 (1970)), the binding
 * energies of the electrons of an ion are approximated by the binding energies of the
 * corresponding electrons of the neutral atom, shifted such that the binding energy of the
 * outermost electron is equal to the ionization energy of the ion.
 */
class ElectronImpactIonizationAtomicData
{
public:
    /**
     * \brief Default constructor of the ElectronImpactIonizationAtomicData class.
     */
    ElectronImpactIonizationAtomicData () = default;

    /**
     * \brief Constructor of the ElectronImpactIonizationAtomicData class. It reads the element
     * of the target ion species of the collision (<ion_species>.physical_element) and the model
     * used for the cross sections (<collision_name>.cross_section_model, where the default
     * value, auto, selects the MBELL model for atomic numbers up to 18 and the Lotz model
     * otherwise).
     *
     * @param[in] collision_name the name of the collision
     * @param[in] mypc pointer to the MultiParticleContainer
     */
    ElectronImpactIonizationAtomicData (std::string const& collision_name,
                                        MultiParticleContainer const * mypc);

    // The pointers in m_data refer to the device vectors owned by this object: copies are
    // therefore forbidden, while moves are allowed (they do not reallocate the data)
    ElectronImpactIonizationAtomicData (ElectronImpactIonizationAtomicData const&) = delete;
    ElectronImpactIonizationAtomicData& operator= (ElectronImpactIonizationAtomicData const&) = delete;
    ElectronImpactIonizationAtomicData (ElectronImpactIonizationAtomicData&&) = default;
    ElectronImpactIonizationAtomicData& operator= (ElectronImpactIonizationAtomicData&&) = default;
    ~ElectronImpactIonizationAtomicData () = default;

    /**
     * \brief Returns the light-weight structure used to access the atomic data in GPU kernels
     */
    [[nodiscard]] ElectronImpactIonization::AtomicData const& data () const { return m_data; }

private:
    amrex::Gpu::DeviceVector<int> m_subshell_offsets;
    amrex::Gpu::DeviceVector<amrex::ParticleReal> m_occupation;
    amrex::Gpu::DeviceVector<amrex::ParticleReal> m_binding_energy;
    amrex::Gpu::DeviceVector<int> m_mbell_subshell;
    amrex::Gpu::DeviceVector<int> m_mbell_q_eff;

    ElectronImpactIonization::AtomicData m_data;
};

#endif // WARPX_ELECTRON_IMPACT_IONIZATION_ATOMIC_DATA_H_
