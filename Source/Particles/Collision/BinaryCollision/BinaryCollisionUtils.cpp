/* Copyright 2021 Neil Zaim
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "BinaryCollisionUtils.H"

#include "Particles/Collision/ScatteringProcess.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"

#include <AMReX_ParmParse.H>
#include <AMReX_Vector.H>

#include <cmath>
#include <string>
#include <sstream>
#include <utility>

#include "Utils/TextMsg.H"

using namespace amrex::literals;

namespace BinaryCollisionUtils{

    CollisionType get_collision_type (const std::string& collision_name,
                                      MultiParticleContainer const * const mypc)
    {
        const amrex::ParmParse pp_collision_name(collision_name);
        // For legacy, pairwisecoulomb is the default
        std::string type = "pairwisecoulomb";
        pp_collision_name.query("type", type);
        if (type == "pairwisecoulomb") {
            return CollisionType::PairwiseCoulomb;
        }
        else if (type == "nuclearfusion") {
            const NuclearFusionType fusion_type = get_nuclear_fusion_type(collision_name, mypc);
            return nuclear_fusion_type_to_collision_type(fusion_type);
        }
        else if (type == "bremsstrahlung") {
            return CollisionType::Bremsstrahlung;
        }
        else if (type == "dsmc") {
            return CollisionType::DSMC;
        }
        else if (type == "electron_impact_ionization") {
            // Check that the species are consistent with this collision type
            get_impact_ionization_species(collision_name, mypc);
            return CollisionType::ElectronImpactIonization;
        }
        else if (type == "ion_impact_ionization") {
            // Check that the species are consistent with this collision type
            get_impact_ionization_species(collision_name, mypc);
            return CollisionType::IonImpactIonization;
        }
        else if (type == "linear_breit_wheeler") {
            amrex::Vector<std::string> species_name;
            // Check that incoming species are photons
            pp_collision_name.getarr("species", species_name);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                species_name.size() == 2u,
                "Linear Breit-Wheeler collisions must involve exactly two species");
            auto& species1 = mypc->GetParticleContainerFromName(species_name[0]);
            auto& species2 = mypc->GetParticleContainerFromName(species_name[1]);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                species1.AmIA<PhysicalSpecies::photon>() && species2.AmIA<PhysicalSpecies::photon>(),
                "Species involved in linear Breit-Wheeler collisions must be of type photon.");
            // Check that product species are electron and positron
            amrex::Vector<std::string> product_species_name;
            pp_collision_name.getarr("product_species", product_species_name);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                product_species_name.size() == 2u,
                "Linear Breit-Wheeler collisions must contain exactly two product species");
            auto& product_species1 = mypc->GetParticleContainerFromName(product_species_name[0]);
            auto& product_species2 = mypc->GetParticleContainerFromName(product_species_name[1]);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                (product_species1.AmIA<PhysicalSpecies::electron>() && product_species2.AmIA<PhysicalSpecies::positron>())
                ||
                (product_species1.AmIA<PhysicalSpecies::positron>() && product_species2.AmIA<PhysicalSpecies::electron>()),
                "Product species of linear Breit-Wheeler collisions must be of type electron and positron");
            return CollisionType::LinearBreitWheeler;
        }
        else if (type == "linear_compton") {
            amrex::Vector<std::string> species_name;
            // Check that the first incoming species is a photon and the second is an electron/positron
            pp_collision_name.getarr("species", species_name);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                species_name.size() == 2u,
                "Linear Compton collisions must involve exactly two species");
            auto& species1 = mypc->GetParticleContainerFromName(species_name[0]);
            auto& species2 = mypc->GetParticleContainerFromName(species_name[1]);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE( species1.AmIA<PhysicalSpecies::photon>(),
                "The first species in linear Compton collisions must be a photon");
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE( species2.AmIA<PhysicalSpecies::electron>() || species2.AmIA<PhysicalSpecies::positron>(),
                "The second species in linear Compton collisions must be an electron or positron");
            // Check that first product species is photon and second is electron/positron
            amrex::Vector<std::string> product_species_name;
            pp_collision_name.getarr("product_species", product_species_name);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                product_species_name.size() == 2u,
                "Linear Compton collisions must contain exactly two product species");
            auto& product_species1 = mypc->GetParticleContainerFromName(product_species_name[0]);
            auto& product_species2 = mypc->GetParticleContainerFromName(product_species_name[1]);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE( product_species1.AmIA<PhysicalSpecies::photon>(),
                "The first product species in linear Compton collisions must be a photon");
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE( product_species2.AmIA<PhysicalSpecies::electron>() || product_species2.AmIA<PhysicalSpecies::positron>(),
                "The second product species in linear Compton collisions must be an electron or positron");
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE( species2.getSpeciesTypeName() == product_species2.getSpeciesTypeName(),
                "The second incident species and the second product species in linear Compton collisions must be of the same type");
            return CollisionType::LinearCompton;
        }
        return CollisionType::Undefined;
    }

    ImpactIonizationSpecies
    get_impact_ionization_species (const std::string& collision_name,
                                   MultiParticleContainer const * const mypc)
    {
        // A species is considered to be an electron species if it was defined with
        // `species_type = electron`, or if it has the charge and mass of an electron
        auto const is_electron_species = [] (WarpXParticleContainer const& species) {
            return species.AmIA<PhysicalSpecies::electron>() ||
                (std::abs(species.getCharge() + PhysConst::q_e) <= 1.e-6_prt*PhysConst::q_e &&
                 std::abs(species.getMass() - PhysConst::m_e) <= 1.e-6_prt*PhysConst::m_e);
        };

        const amrex::ParmParse pp_collision_name(collision_name);
        std::string type;
        pp_collision_name.get("type", type);
        ImpactIonizationSpecies result;
        result.ion_projectile = (type == "ion_impact_ionization");
        std::string const description =
            (result.ion_projectile ? "Ion-impact" : "Electron-impact") +
            std::string(" ionization collision ") + collision_name;

        amrex::Vector<std::string> species_names;
        pp_collision_name.getarr("species", species_names);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            species_names.size() == 2u,
            description + " must involve exactly two species: the projectile and the target");

        // Identify the target: this is the species that carries an ionization level, if any,
        // and the second species otherwise
        result.projectile = species_names[0];
        result.target = species_names[1];
        if (mypc->GetParticleContainerFromName(result.projectile).DoFieldIonization()) {
            std::swap(result.projectile, result.target);
        }
        auto const& projectile = mypc->GetParticleContainerFromName(result.projectile);
        auto const& target = mypc->GetParticleContainerFromName(result.target);
        result.atomic_target = target.DoFieldIonization();
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !projectile.DoFieldIonization(),
            description + ": only one of the two species can have field ionization enabled "
            "(the projectile cannot have an ionization level)");
        if (result.ion_projectile) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                projectile.getCharge() > 0._prt,
                description + ": the projectile species " + result.projectile +
                " must be positively charged (when the target does not have field ionization "
                "enabled, the projectile species must be the first species)");
            // The model of the cross sections and of the kinematics is designed for heavy
            // projectiles (the ionization by electrons is modeled by electron_impact_ionization)
            constexpr amrex::ParticleReal min_mass_ratio = 100._prt;
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                projectile.getMass() > min_mass_ratio*PhysConst::m_e,
                description + ": the projectile species " + result.projectile +
                " must be an ion species (with a mass larger than 100 electron masses)");
        } else {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                result.atomic_target,
                description + ": exactly one of the two species must be an ion species with "
                "<species_name>.do_field_ionization = 1, which defines its element "
                "and its ionization level");
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                is_electron_species(projectile),
                description + ": the species " + result.projectile +
                " must be an electron species");
        }

        amrex::Vector<std::string> product_species_names;
        pp_collision_name.queryarr("product_species", product_species_names);
        if (result.atomic_target) {
            // The ionization level of the target is increased, and the ejected electrons are
            // added to the product species if specified, and otherwise to the product species
            // of the field ionization of the target
            result.product_ion = result.target;
            if (product_species_names.empty()) {
                const amrex::ParmParse pp_target_name(result.target);
                pp_target_name.get("ionization_product_species", result.product_electron);
            } else {
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    product_species_names.size() == 1u,
                    description + " must have exactly one product species, for the ejected "
                    "electrons, since the target has an ionization level");
                result.product_electron = product_species_names[0];
            }
        } else {
            // The ionized targets are converted to the product ion species
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                product_species_names.size() == 2u,
                description + " must have exactly two product species, for the ionized "
                "targets and for the ejected electrons, since the target species " +
                result.target + " does not have an ionization level");
            result.product_electron = product_species_names[0];
            result.product_ion = product_species_names[1];
            if (!is_electron_species(mypc->GetParticleContainerFromName(result.product_electron))) {
                std::swap(result.product_electron, result.product_ion);
            }
            auto const& product_ion = mypc->GetParticleContainerFromName(result.product_ion);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                std::abs(product_ion.getCharge() - target.getCharge() - PhysConst::q_e)
                <= 1.e-6_prt*PhysConst::q_e,
                description + ": the charge of the product species " + result.product_ion +
                " must be larger than the charge of the target species " + result.target +
                " by one elementary charge");
        }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            is_electron_species(mypc->GetParticleContainerFromName(result.product_electron)),
            description + ": the product species " + result.product_electron +
            " must be an electron species");

        return result;
    }

    amrex::Vector<ScatteringProcess>
    parse_scattering_processes (const std::string& collision_name)
    {
        using namespace amrex::literals;

        const amrex::ParmParse pp_collision_name(collision_name);

        amrex::Vector<std::string> scattering_process_names;
        pp_collision_name.queryarr("scattering_processes", scattering_process_names);

        // create a vector of ScatteringProcess objects from each scattering
        // process name
        amrex::Vector<ScatteringProcess> scattering_processes;
        for (const auto& scattering_process : scattering_process_names) {
            const std::string kw_cross_section = scattering_process + "_cross_section";
            std::string cross_section_file;
            pp_collision_name.query(kw_cross_section, cross_section_file);

            const auto process_type = ScatteringProcess::parseProcessType(scattering_process);

            // The energy cost (penalty) of the process, in eV. It is required for excitation
            // and ionization, optional for charge exchange and two-product reactions (which may
            // impose a fixed energy loss), and not read for elastic processes.
            amrex::ParticleReal energy = 0._prt;
            const std::string kw_energy = scattering_process + "_energy";
            if (process_type == ScatteringProcessType::EXCITATION ||
                process_type == ScatteringProcessType::IONIZATION) {
                utils::parser::getWithParser(
                    pp_collision_name, kw_energy.c_str(), energy);
            } else if (process_type != ScatteringProcessType::ELASTIC) {
                utils::parser::queryWithParser(
                    pp_collision_name, kw_energy.c_str(), energy);
            }

            // The angular behavior of a process is controlled by the per-process
            // `<process>_scattering_angle_model` argument.
            // The default angle model depends on the process: product-producing processes
            // (charge exchange and two-product reactions) default to forward scattering, while
            // particle-conserving processes (e.g. elastic, excitation) default to isotropic.
            auto scattering_angle_model =
                (process_type == ScatteringProcessType::CHARGE_EXCHANGE ||
                 process_type == ScatteringProcessType::TWO_PRODUCT_REACTION)
                ? ScatteringAngleModel::Forward : ScatteringAngleModel::Isotropic;
            pp_collision_name.query_enum_case_insensitive(
                scattering_process + "_scattering_angle_model", scattering_angle_model);

            // DSMC/MCC collisions do not support an angular-distribution coefficient
            // table (unlike nuclear fusion), so anisotropic scattering cannot be honored here.
            std::string scattering_angle_error_msg = collision_name;
            scattering_angle_error_msg += ".";
            scattering_angle_error_msg += scattering_process;
            scattering_angle_error_msg += "_scattering_angle_model = legendre is not supported for DSMC/MCC collisions.";
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(scattering_angle_model != ScatteringAngleModel::Legendre, scattering_angle_error_msg);
            scattering_processes.push_back(ScatteringProcess(
                scattering_process, cross_section_file, energy, scattering_angle_model));
        }

        return scattering_processes;
    }

    NuclearFusionType get_nuclear_fusion_type (const std::string& collision_name,
                                               MultiParticleContainer const * const mypc)
    {
        using namespace amrex::literals;

        const amrex::ParmParse pp_collision_name(collision_name);
        amrex::Vector<std::string> species_names;
        pp_collision_name.getarr("species", species_names);
        auto& species1 = mypc->GetParticleContainerFromName(species_names[0]);
        auto& species2 = mypc->GetParticleContainerFromName(species_names[1]);

        // Compute the total mass before and after collision to check the fusion energy
        const amrex::ParticleReal mass_before = species1.getMass() + species2.getMass();
        amrex::ParticleReal mass_after = 0.0_prt;

        amrex::Vector<std::string> product_species_name;
        pp_collision_name.getarr("product_species", product_species_name);

        NuclearFusionType fusion_type = NuclearFusionType::Undefined;
        if ((species1.AmIA<PhysicalSpecies::hydrogen2>() && species2.AmIA<PhysicalSpecies::hydrogen3>())
            ||
            (species1.AmIA<PhysicalSpecies::hydrogen3>() && species2.AmIA<PhysicalSpecies::hydrogen2>())
            )
        {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                product_species_name.size() == 2u,
                "ERROR: Deuterium-tritium fusion must contain exactly two product species");
            auto& product_species1 = mypc->GetParticleContainerFromName(product_species_name[0]);
            auto& product_species2 = mypc->GetParticleContainerFromName(product_species_name[1]);
            mass_after = product_species1.getMass() + product_species2.getMass();
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                (product_species1.AmIA<PhysicalSpecies::helium4>() && product_species2.AmIA<PhysicalSpecies::neutron>())
                ||
                (product_species1.AmIA<PhysicalSpecies::neutron>() && product_species2.AmIA<PhysicalSpecies::helium4>()),
                "ERROR: Product species of deuterium-tritium fusion must be of type neutron and helium4");
            fusion_type = NuclearFusionType::DeuteriumTritiumToNeutronHelium;
        }
        else if (species1.AmIA<PhysicalSpecies::hydrogen2>() && species2.AmIA<PhysicalSpecies::hydrogen2>())
        {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                product_species_name.size() == 2u,
                "ERROR: Deuterium-deuterium fusion must contain exactly two product species");
            auto& product_species1 = mypc->GetParticleContainerFromName(product_species_name[0]);
            auto& product_species2 = mypc->GetParticleContainerFromName(product_species_name[1]);
            mass_after = product_species1.getMass() + product_species2.getMass();
            if (
                (product_species1.AmIA<PhysicalSpecies::helium3>() && product_species2.AmIA<PhysicalSpecies::neutron>())
                ||(product_species1.AmIA<PhysicalSpecies::neutron>() && product_species2.AmIA<PhysicalSpecies::helium3>())){
                fusion_type = NuclearFusionType::DeuteriumDeuteriumToNeutronHelium;
            } else if (
                (product_species1.AmIA<PhysicalSpecies::hydrogen3>() && product_species2.AmIA<PhysicalSpecies::proton>())
                ||(product_species1.AmIA<PhysicalSpecies::proton>() && product_species2.AmIA<PhysicalSpecies::hydrogen3>())){
                fusion_type = NuclearFusionType::DeuteriumDeuteriumToProtonTritium;
            } else {
                WARPX_ABORT_WITH_MESSAGE("ERROR: Product species of deuterium-deuterium fusion must be of type helium3 and neutron, or tritium/hydrogen3 and proton/hydrogen1");
            }
        }
        else if ((species1.AmIA<PhysicalSpecies::hydrogen2>() && species2.AmIA<PhysicalSpecies::helium3>())
            ||
            (species1.AmIA<PhysicalSpecies::helium3>() && species2.AmIA<PhysicalSpecies::hydrogen2>())
            )
        {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                product_species_name.size() == 2u,
                "ERROR: Deuterium-helium fusion must contain exactly two product species");
            auto& product_species1 = mypc->GetParticleContainerFromName(product_species_name[0]);
            auto& product_species2 = mypc->GetParticleContainerFromName(product_species_name[1]);
            mass_after = product_species1.getMass() + product_species2.getMass();
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                (product_species1.AmIA<PhysicalSpecies::helium4>() && product_species2.AmIA<PhysicalSpecies::proton>())
                ||
                (product_species1.AmIA<PhysicalSpecies::proton>() && product_species2.AmIA<PhysicalSpecies::helium4>()),
                "ERROR: Product species of deuterium-helium fusion must be of type proton/hydrogen1 and alpha/helium4");
            fusion_type = NuclearFusionType::DeuteriumHeliumToProtonHelium;
        }
        else if ((species1.AmIA<PhysicalSpecies::proton>() && species2.AmIA<PhysicalSpecies::boron11>())
            ||
            (species1.AmIA<PhysicalSpecies::boron11>() && species2.AmIA<PhysicalSpecies::proton>())
            )
        {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                product_species_name.size() == 1,
                "ERROR: Proton-boron must contain exactly one product species");
            auto& product_species = mypc->GetParticleContainerFromName(product_species_name[0]);
            mass_after = 3.0_prt*product_species.getMass();
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                product_species.AmIA<PhysicalSpecies::helium4>(),
                "ERROR: Product species of proton-boron fusion must be of type helium4");
            fusion_type = NuclearFusionType::ProtonBoronToAlphas;
        }
        else {
            WARPX_ABORT_WITH_MESSAGE("Binary nuclear fusion not implemented between species " +
                            species_names[0] + " of type " + species1.getSpeciesTypeName() +
                            " and species " + species_names[1] + " of type " +
                            species2.getSpeciesTypeName());
        }

        CheckFusionEnergy(fusion_type, mass_before, mass_after);

        return fusion_type;
    }

    void CheckFusionEnergy (const NuclearFusionType fusion_type,
                            const amrex::ParticleReal mass_before,
                            const amrex::ParticleReal mass_after)
    {
        // Compute the fusion energy
        const amrex::ParticleReal fusion_energy = (mass_before - mass_after)*PhysConst::c2;

        // Verify that the fusion energy is close to what is expected.
        // The expected fusion energies are computed using fully ionized species mass
        // computed as M = A*m_u - Z*m_e, with the atomic mass numbers A found in
        // SpeciesPhysicalProperties.cpp
        std::ostringstream error_msg;
        amrex::ParticleReal expected_fusion_energy = 0.0_prt;
        if (fusion_type == NuclearFusionType::DeuteriumTritiumToNeutronHelium) {
            expected_fusion_energy = 17.58929696e6_prt * PhysConst::q_e;
            error_msg << "Fusion energy mismatch in D + T -> He4 + n\n";
        }
        if (fusion_type == NuclearFusionType::DeuteriumDeuteriumToProtonTritium) {
            expected_fusion_energy = 4.032653948e6_prt * PhysConst::q_e;
            error_msg << "Fusion energy mismatch in D + D -> T + p\n";
        }
        if (fusion_type == NuclearFusionType::DeuteriumDeuteriumToNeutronHelium) {
            expected_fusion_energy = 3.26891111e6_prt * PhysConst::q_e;
            error_msg << "Fusion energy mismatch in D + D -> He3 + n\n";
        }
        if (fusion_type == NuclearFusionType::DeuteriumHeliumToProtonHelium) {
            expected_fusion_energy = 18.35303980e6_prt * PhysConst::q_e;
            error_msg << "Fusion energy mismatch in D + He3 -> He3 + p\n";
        }
        if (fusion_type == NuclearFusionType::ProtonBoronToAlphas) {
            expected_fusion_energy = 8.68212502e6_prt * PhysConst::q_e;
            error_msg << "Fusion energy mismatch in p + B11 -> 3 He4\n";
        }

        const amrex::ParticleReal energy_error = amrex::Math::abs(fusion_energy - expected_fusion_energy);
        const amrex::ParticleReal energy_rel_tol = 0.01_prt;

        error_msg << "  energy error [eV]           = " << energy_error / PhysConst::q_e << "\n"
                  << "  energy tolerance (rel.)     = " << energy_rel_tol << "\n"
                  << "  expected fusion energy [eV] = " << expected_fusion_energy / PhysConst::q_e << "\n"
                  << "  computed fusion energy [eV] = " << fusion_energy / PhysConst::q_e<< "\n"
                  << "Check that species masses are set correctly.";

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            energy_error < energy_rel_tol*expected_fusion_energy,
            error_msg.str()
        );

    }
}
