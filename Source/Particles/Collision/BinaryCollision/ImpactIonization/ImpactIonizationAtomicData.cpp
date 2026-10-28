/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "ImpactIonizationAtomicData.H"

#include "Particles/Collision/BinaryCollision/BinaryCollisionUtils.H"
#include "Particles/Collision/ScatteringProcess.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/Physics/BindingEnergiesTable.H"
#include "Utils/Physics/IonizationEnergiesTable.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"

#include <AMReX_GpuContainers.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Vector.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace
{
    using RuddParameters = std::array<double, ImpactIonization::n_rudd_parameters>;

    // Parameters of the Rudd model (A1, B1, C1, D1, E1, A2, B2, C2, D2, alpha), from Table V of
    // Rudd et al., Rev. Mod. Phys. 64, 441 (1992), for the outer subshells of the rare gases and
    // for the inner subshells of all targets
    constexpr RuddParameters rudd_helium =
        {1.02, 2.4, 0.70, 1.15, 0.70, 0.84, 6.0, 0.70, 0.50, 0.86};
    constexpr RuddParameters rudd_neon =
        {0.58, 65.0, 0.23, 0.55, 0.16, 1.40, 0.0, 0.72, 1.35, 0.57};
    constexpr RuddParameters rudd_argon =
        {1.20, 8.0, 0.86, 0.0, 0.80, 0.90, 2.7, 0.75, 0.80, 0.71};
    constexpr RuddParameters rudd_krypton =
        {1.46, 5.7, 0.65, -0.55, 1.00, 1.30, 22.0, 0.95, -1.00, 0.78};
    constexpr RuddParameters rudd_inner_subshells =
        {1.25, 0.50, 1.00, 1.00, 3.0, 1.10, 1.30, 1.00, 0.0, 0.66};
    // One-electron targets, for which Rudd et al. give no parameters: parameters of helium,
    // except A1 = 0.75 and A2 = 1.2. A2 sets the cross section for fast ejected electrons
    // (A2 times the Rutherford cross section), and is close to 1 as for the targets of Rudd et
    // al. (1992). A1 was then adjusted to the cross sections of hydrogen recommended by Rudd et
    // al., Rev. Mod. Phys. 57, 965 (1985), with which the model agrees within 7% from 50 keV to
    // 100 MeV. (The Bethe values A1 = 2 M^2 = 0.567 and A2 = 2 would reproduce these cross
    // sections, but with twice the Rutherford cross section for fast electrons, and an energy
    // lost in ionization larger than the stopping power.)
    constexpr RuddParameters rudd_hydrogen =
        {0.75, 2.4, 0.70, 1.15, 0.70, 1.2, 6.0, 0.70, 0.50, 0.86};

    /**
     * \brief Default parameters of the Rudd model for the outer subshells of an atom or ion, as a
     * function of its number of bound electrons. The parameters of the rare gas with the same
     * number of electrons are used for the closed-shell configurations (He-like, Ne-like,
     * Ar-like and Kr-like), and otherwise the parameters of the rare gas of the row of the
     * periodic table that contains the outer electrons, except for the second row, for which the
     * parameters of helium are used: those of neon have a small dipole parameter A1 (a compromise
     * for the fit of the data of neon, see Sec. X.B of Rudd et al. 1992), which would give a
     * dipole matrix element smaller than half that of N2 or O2 for the atoms N and O. The
     * parameters of krypton are used for all the targets with more than 18 electrons.
     *
     * @param[in] n_electrons number of bound electrons of the atom or ion
     * @return the parameters of the outer subshells
     */
    RuddParameters DefaultOuterSubshellParameters (int const n_electrons)
    {
        if (n_electrons == 1) { return rudd_hydrogen; }
        if (n_electrons < 10) { return rudd_helium; }
        if (n_electrons == 10) { return rudd_neon; }
        if (n_electrons <= 18) { return rudd_argon; }
        return rudd_krypton;
    }

    /**
     * \brief Cross section of a subshell in the Rudd model, i.e. integral of the singly
     * differential cross section of the model (see ImpactIonization::RuddModel) over the kinetic
     * energy W of the ejected electron, from 0 to E - B, for a projectile of charge number 1.
     * The integral is computed with the variable y = w/(1+w), in which the integrand is
     * (F1 (1-y) + F2 y)/(1 + exp(alpha (w - w_c)/v)): the interval of integration is split at
     * points spaced by 8 times the width v/alpha of the cutoff around w_c (where the integrand
     * varies rapidly), and in intervals of at most 1/8 in y, and each interval is integrated with
     * a 16-point Gauss-Legendre quadrature. The relative accuracy is better than 1e-6.
     *
     * @param[in] E kinetic energy of the projectile [eV]
     * @param[in] M rest energy of the projectile [eV]
     * @param[in] B binding energy of the subshell [eV]
     * @param[in] N number of electrons in the subshell
     * @param[in] p parameters of the Rudd model of the subshell
     * @return the cross section [m^2]
     */
    double RuddSubshellCrossSection (double const E, double const M, double const B,
                                     double const N, double const* p)
    {
        using ImpactIonization::m_e_c2_eV;
        if (E <= B) { return 0.0; }

        // Nodes and weights of the 16-point Gauss-Legendre quadrature (positive nodes)
        constexpr int n_nodes = 8;
        constexpr std::array<double, n_nodes> nodes = {
            0.0950125098376374, 0.2816035507792589, 0.4580167776572274, 0.6178762444026438,
            0.7554044083550030, 0.8656312023878318, 0.9445750230732326, 0.9894009349916499};
        constexpr std::array<double, n_nodes> weights = {
            0.1894506104550686, 0.1826034150449236, 0.1691565193950026, 0.1495959888165767,
            0.1246289712555341, 0.0951585116824926, 0.0622535239386475, 0.0271524594117542};

        double F1, F2, v, w_c;
        ImpactIonization::RuddModel(E, M, B, p, F1, F2, v, w_c);
        double const alpha = p[9];
        double const width = v/alpha;
        // Upper bound of w: energy conservation, and the energy above which the cutoff factor is
        // smaller than exp(-40)
        double const w_up = std::min((E - B)/B, std::max(w_c, 0.0) + 40.0*width);
        double const y_up = w_up/(1.0 + w_up);

        std::vector<double> breaks = {0.0, y_up};
        for (int k = -3; k <= 5; ++k) {
            double const w = w_c + 8.0*k*width;
            if (w > 0.0 && w < w_up) { breaks.push_back(w/(1.0 + w)); }
        }
        std::sort(breaks.begin(), breaks.end());

        auto const integrand = [&] (double const y) {
            double const w = y/(1.0 - y);
            double const arg = alpha*(w - w_c)/v;
            double const cutoff = (arg > 700.0) ? 0.0 : 1.0/(1.0 + std::exp(arg));
            return (F1*(1.0 - y) + F2*y)*cutoff;
        };
        double integral = 0.0;
        for (std::size_t i = 0; i + 1 < breaks.size(); ++i) {
            double const length = breaks[i+1] - breaks[i];
            int const n_intervals = std::max(1, static_cast<int>(std::ceil(8.0*length)));
            double const h = length/n_intervals;
            for (int j = 0; j < n_intervals; ++j) {
                double const center = breaks[i] + (j + 0.5)*h;
                for (int k = 0; k < n_nodes; ++k) {
                    integral += 0.5*h*weights[k]*(integrand(center - 0.5*h*nodes[k]) +
                                                  integrand(center + 0.5*h*nodes[k]));
                }
            }
        }

        // S = 4 pi a0^2 N (R/B)^2, written as pi r_e^2 N (m_e c^2/B)^2
        constexpr double r_e = PhysConst::r_e_v<double>;
        double const S = MathConst::pi_v<double>*r_e*r_e*N*(m_e_c2_eV/B)*(m_e_c2_eV/B);
        return S*integral;
    }
}

ImpactIonizationAtomicData::ImpactIonizationAtomicData (
    std::string const& collision_name, MultiParticleContainer const * const mypc)
{
    static_assert(utils::physics::binding_energies_tab_length
                  == utils::physics::energies_tab_length,
                  "The tables of binding and ionization energies must have the same layout");

    auto const species = BinaryCollisionUtils::get_impact_ionization_species(collision_name, mypc);
    const amrex::ParmParse pp_collision_name(collision_name);
    std::string const description =
        (species.ion_projectile ? "Ion-impact" : "Electron-impact") +
        std::string(" ionization collision ") + collision_name;

    // Occupied subshells of each charge state of the target, from the least bound to the most
    // bound one, so that the binding energies are in increasing order for each charge state
    int atomic_number = 0;
    amrex::Vector<int> h_subshell_offsets;
    amrex::Vector<amrex::ParticleReal> h_occupation;
    amrex::Vector<amrex::ParticleReal> h_binding_energy;
    amrex::Vector<int> h_mbell_subshell;
    amrex::Vector<int> h_mbell_q_eff;
    auto model = IonizationCrossSectionModel::Default;

    // The MBELL model is defined for the subshells 1s, 2s, 2p, 3s and 3p
    constexpr int max_mbell_atomic_number = 18;
    constexpr std::array<int, ImpactIonization::n_mbell_subshells> mbell_capacity =
        {2, 2, 6, 2, 6};

    // Check that the parameters specified by the user are consistent with the collision
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !species.ion_projectile || !pp_collision_name.contains("cross_section_model"),
        description + ": " + collision_name + ".cross_section_model is only used for "
        "electron-impact ionization");
    for (char const* name : {"binding_energies", "occupations", "cross_section"}) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !species.atomic_target || !pp_collision_name.contains(name),
            description + ": " + collision_name + "." + name + " can only be used for a "
            "molecular target (i.e. a target species without field ionization)");
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        species.ion_projectile || !pp_collision_name.contains("rudd_parameters"),
        description + ": " + collision_name + ".rudd_parameters is only used for ion-impact "
        "ionization");

    if (species.atomic_target) {
        // The element of the target is given by physical_element
        std::string physical_element;
        const amrex::ParmParse pp_target_name(species.target);
        pp_target_name.get("physical_element", physical_element);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            utils::physics::ion_map_ids.count(physical_element) == 1,
            description + ": unknown physical element " + physical_element);
        int const element_id = utils::physics::ion_map_ids.at(physical_element);
        atomic_number = utils::physics::ion_atomic_numbers[element_id];
        int const offset = utils::physics::ion_energy_offsets[element_id];
        auto const* const ionization_energy = utils::physics::table_ionization_energies + offset;
        auto const* const neutral_binding_energy =
            utils::physics::table_neutral_binding_energies + offset;

        if (!species.ion_projectile) {
            pp_collision_name.query_enum_sloppy("cross_section_model", model, "-_");
            if (model == IonizationCrossSectionModel::Auto) {
                model = (atomic_number <= max_mbell_atomic_number) ?
                    IonizationCrossSectionModel::MBELL : IonizationCrossSectionModel::Lotz;
            }
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                model != IonizationCrossSectionModel::MBELL ||
                atomic_number <= max_mbell_atomic_number,
                description + ": the MBELL model is only available for atomic numbers up to " +
                std::to_string(max_mbell_atomic_number));
        }

        h_subshell_offsets.resize(atomic_number + 1);
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
                    n_subshells <= ImpactIonization::n_mbell_subshells,
                    description + ": unexpected number of subshells");
                for (int k = 0; k < n_subshells; ++k) {
                    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(occupation[k] <= mbell_capacity[k],
                        description + ": unexpected occupation of a subshell");
                }
            }

            // Shift of the binding energies, such that the binding energy of the outermost
            // electron is the ionization energy of the ion (Carlson et al., 1970)
            double const shift = static_cast<double>(ionization_energy[charge_state])
                                 - binding_energy.back();

            int n_electrons_up_to_subshell = n_electrons;
            for (int k = n_subshells - 1; k >= 0; --k) {
                h_occupation.push_back(static_cast<amrex::ParticleReal>(occupation[k]));
                h_binding_energy.push_back(
                    static_cast<amrex::ParticleReal>(binding_energy[k] + shift));
                h_mbell_subshell.push_back(atomic_number <= max_mbell_atomic_number ? k : -1);
                h_mbell_q_eff.push_back(atomic_number - n_electrons_up_to_subshell);
                n_electrons_up_to_subshell -= occupation[k];
            }
        }
        h_subshell_offsets[atomic_number] = static_cast<int>(h_occupation.size());
    } else {
        // Molecular target, which can be ionized once: the binding energies and the occupations
        // of its orbitals are given by the user
        std::vector<double> binding_energies;
        std::vector<double> occupations;
        utils::parser::getArrWithParser(pp_collision_name, "binding_energies", binding_energies);
        utils::parser::queryArrWithParser(pp_collision_name, "occupations", occupations);
        if (occupations.empty() && binding_energies.size() == 1u) { occupations = {1.0}; }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            occupations.size() == binding_energies.size(),
            description + ": " + collision_name + ".occupations must have the same number of "
            "values as " + collision_name + ".binding_energies");
        std::vector<std::size_t> order(binding_energies.size());
        for (std::size_t i = 0; i < order.size(); ++i) { order[i] = i; }
        std::sort(order.begin(), order.end(), [&] (std::size_t a, std::size_t b) {
            return binding_energies[a] < binding_energies[b]; });
        atomic_number = 1;
        h_subshell_offsets = {0, static_cast<int>(order.size())};
        for (auto const i : order) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(binding_energies[i] > 0.0 && occupations[i] > 0.0,
                description + ": the binding energies and the occupations must be positive");
            h_occupation.push_back(static_cast<amrex::ParticleReal>(occupations[i]));
            h_binding_energy.push_back(static_cast<amrex::ParticleReal>(binding_energies[i]));
            h_mbell_subshell.push_back(-1);
            h_mbell_q_eff.push_back(0);
        }
    }

    // Transfer the data to the GPU device
    auto const copy_to_device = [] (auto const& host_vector, auto& device_vector) {
        device_vector.resize(host_vector.size());
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, host_vector.begin(), host_vector.end(),
                              device_vector.begin());
    };
    copy_to_device(h_subshell_offsets, m_subshell_offsets);
    copy_to_device(h_occupation, m_occupation);
    copy_to_device(h_binding_energy, m_binding_energy);
    copy_to_device(h_mbell_subshell, m_mbell_subshell);
    copy_to_device(h_mbell_q_eff, m_mbell_q_eff);

    m_data.m_atomic_number = atomic_number;
    m_data.m_model = model;
    m_data.m_subshell_offsets = m_subshell_offsets.dataPtr();
    m_data.m_occupation = m_occupation.dataPtr();
    m_data.m_binding_energy = m_binding_energy.dataPtr();
    m_data.m_mbell_subshell = m_mbell_subshell.dataPtr();
    m_data.m_mbell_q_eff = m_mbell_q_eff.dataPtr();

    if (species.ion_projectile) {
        auto const& projectile = mypc->GetParticleContainerFromName(species.projectile);
        // Rest energy [eV] and charge number of the projectile
        double const M = static_cast<double>(projectile.getMass())*PhysConst::c2_v<double>
                         /PhysConst::q_e_v<double>;
        double const Zp = static_cast<double>(projectile.getCharge())/PhysConst::q_e_v<double>;
        m_data.m_projectile = ImpactIonization::ProjectileType::Ion;
        m_data.m_projectile_mass_eV = M;

        // Parameters of the outer subshells given by the user, if any
        std::vector<double> user_parameters;
        utils::parser::queryArrWithParser(pp_collision_name, "rudd_parameters", user_parameters);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            user_parameters.empty() ||
            static_cast<int>(user_parameters.size()) == ImpactIonization::n_rudd_parameters,
            description + ": " + collision_name + ".rudd_parameters must contain 10 values "
            "(A1 B1 C1 D1 E1 A2 B2 C2 D2 alpha)");

        // Parameters of each subshell. Following Rudd et al. (1992), the inner subshells are those
        // whose binding energy exceeds twice that of the least bound subshell.
        amrex::Vector<double> h_rudd_parameters;
        for (int charge_state = 0; charge_state < atomic_number; ++charge_state) {
            int const first = h_subshell_offsets[charge_state];
            int const end = h_subshell_offsets[charge_state+1];
            RuddParameters outer = rudd_helium;
            if (!user_parameters.empty()) {
                std::copy(user_parameters.begin(), user_parameters.end(), outer.begin());
            } else if (species.atomic_target) {
                outer = DefaultOuterSubshellParameters(atomic_number - charge_state);
            }
            for (int s = first; s < end; ++s) {
                bool const inner = (h_binding_energy[s] > 2*h_binding_energy[first]);
                RuddParameters const& p = inner ? rudd_inner_subshells : outer;
                h_rudd_parameters.insert(h_rudd_parameters.end(), p.begin(), p.end());
            }
        }

        // Tables of the cross sections, as a function of the momentum per unit mass gamma*beta
        // of the projectile, from 1e-4 to 1e4 with 64 points per decade
        constexpr double u_min = 1.e-4;
        constexpr int points_per_decade = 64;
        constexpr int n_decades = 8;
        int const table_size = n_decades*points_per_decade + 1;
        double const dlog_u = std::log(10.0)/points_per_decade;
        // Logarithm of the cross sections [m^2], with a floor for vanishing cross sections
        constexpr double sigma_floor = 1.e-200;
        auto const n_subshells = static_cast<int>(h_binding_energy.size());
        amrex::Vector<double> h_subshell_table(static_cast<std::size_t>(n_subshells)*table_size);
        amrex::Vector<double> h_total_table(static_cast<std::size_t>(atomic_number)*table_size);
#ifdef AMREX_USE_OMP
#pragma omp parallel for
#endif
        for (int charge_state = 0; charge_state < atomic_number; ++charge_state) {
            for (int i = 0; i < table_size; ++i) {
                double const u = u_min*std::exp(i*dlog_u);
                // Kinetic energy of the projectile, written to avoid cancellations
                double const E = M*u*u/(std::sqrt(1.0 + u*u) + 1.0);
                double total = 0.0;
                for (int s = h_subshell_offsets[charge_state];
                     s < h_subshell_offsets[charge_state+1]; ++s) {
                    double const sigma = Zp*Zp*RuddSubshellCrossSection(E, M,
                        static_cast<double>(h_binding_energy[s]),
                        static_cast<double>(h_occupation[s]),
                        h_rudd_parameters.data() + s*ImpactIonization::n_rudd_parameters);
                    total += sigma;
                    h_subshell_table[s*table_size + i] = std::log(std::max(sigma, sigma_floor));
                }
                h_total_table[charge_state*table_size + i] =
                    std::log(std::max(total, sigma_floor));
            }
        }
        copy_to_device(h_rudd_parameters, m_rudd_parameters);
        copy_to_device(h_subshell_table, m_subshell_table);
        copy_to_device(h_total_table, m_total_table);
        // The host vectors must not be destroyed before the end of the copies
        amrex::Gpu::streamSynchronize();
        m_data.m_rudd_parameters = m_rudd_parameters.dataPtr();
        m_data.m_subshell_table = m_subshell_table.dataPtr();
        m_data.m_total_table = m_total_table.dataPtr();
        m_data.m_table_size = table_size;
        m_data.m_table_log_u_min = std::log(u_min);
        m_data.m_table_inv_dlog_u = 1.0/dlog_u;

        // For a molecular target, the total cross section can be read from a file, which has the
        // same format as for the other collision types: two columns, the kinetic energy of the
        // projectile in the rest frame of the target [eV] and the cross section [m^2]
        std::string cross_section_file;
        if (!species.atomic_target &&
            pp_collision_name.query("cross_section", cross_section_file)) {
            m_user_cross_section = std::make_unique<ScatteringProcess>(
                "ionization", cross_section_file, amrex::ParticleReal(0.),
                ScatteringAngleModel::Isotropic);
            m_data.m_use_user_cross_section = true;
            m_data.m_user_cross_section = m_user_cross_section->executor();
        }
    }
    amrex::Gpu::streamSynchronize();
}
