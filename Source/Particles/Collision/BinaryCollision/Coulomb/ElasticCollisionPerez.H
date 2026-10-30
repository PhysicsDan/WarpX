/* Copyright 2019 Yinjian Zhao
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLES_COLLISION_ELASTIC_COLLISION_PEREZ_H_
#define WARPX_PARTICLES_COLLISION_ELASTIC_COLLISION_PEREZ_H_

#include "UpdateMomentumPerezElastic.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/WarpXConst.H"

#include <AMReX_Random.H>

/** Prepare information for and call UpdateMomentumPerezElastic().
 *
 * @tparam T_index type of index arguments
 * @tparam T_PR type of particle related floating point arguments
 * @tparam T_R type of other floating point arguments
 * @tparam SoaData_type type of the "struct of array" for the two involved species
 * @param[in] I1s,I2s is the start index for I1,I2 (inclusive).
 * @param[in] I1e,I2e is the stop index for I1,I2 (exclusive).
 * @param[in] I1,I2 the index arrays. They determine all elements that will be used.
 * @param[in,out] soa_1,soa_2 the struct of array for species 1/2
 * @param[in] n1,n2 density of species 1/2
 * @param[in] T1,T2 temperature [Joules] of species 1/2
 *            only used if L <= 0
 * @param[in] q1,q2 charge of species 1/2 (for a species with field ionization, the charge of each
 *            macroparticle is q1/q2 times its ionization level)
 * @param[in] m1,m2 mass of species 1/2
 * @param[in] dt is the time step length between two collision calls.
 * @param[in] global_lamdb is the global Debye length and will be used if greater than zero
 * @param[in] scattering_model is the model used to obtain the scattering angle
 * @param[in] L is the Coulomb log and will be used if greater than zero,
 *            otherwise will be computed.
 * @param[in] dV is the volume of the corresponding cell.
 * @param[in] engine the random number generator state & factory
 * @param[in] isSameSpecies whether this is an intra-species collision process
 * @param[in] coll_idx is the collision index offset.
 * @param[in] ionization_level_comp1,ionization_level_comp2 index of the ionization level among the
 *            runtime integer attributes of species 1/2, or -1 if the species does not have
 *            field ionization
*/

template <typename T_index, typename T_PR, typename T_R, typename SoaData_type>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void ElasticCollisionPerez (
    T_index const I1s, T_index const I1e,
    T_index const I2s, T_index const I2e,
    T_index const* AMREX_RESTRICT I1,
    T_index const* AMREX_RESTRICT I2,
    SoaData_type soa_1, SoaData_type soa_2,
    T_PR const  n1, T_PR const  n2,
    T_PR const  T1, T_PR const  T2,
    T_PR const  q1, T_PR const  q2,
    T_PR const  m1, T_PR const  m2,
    T_R const  dt, T_R const global_lamdb,
    CumulativeScatteringModel const scattering_model,
    T_PR const  L, T_R const  dV,
    amrex::RandomEngine const& engine,
    bool const isSameSpecies, T_index coll_idx,
    int const ionization_level_comp1 = -1, int const ionization_level_comp2 = -1)
{
    const T_index NI1 = I1e - I1s;
    const T_index NI2 = I2e - I2s;
    const T_index max_N = amrex::max(NI1,NI2);
    const T_index min_N = amrex::min(NI1,NI2);

    T_PR * const AMREX_RESTRICT w1 = soa_1.m_rdata[PIdx::w];
    T_PR * const AMREX_RESTRICT u1x = soa_1.m_rdata[PIdx::ux];
    T_PR * const AMREX_RESTRICT u1y = soa_1.m_rdata[PIdx::uy];
    T_PR * const AMREX_RESTRICT u1z = soa_1.m_rdata[PIdx::uz];

    T_PR * const AMREX_RESTRICT w2 = soa_2.m_rdata[PIdx::w];
    T_PR * const AMREX_RESTRICT u2x = soa_2.m_rdata[PIdx::ux];
    T_PR * const AMREX_RESTRICT u2y = soa_2.m_rdata[PIdx::uy];
    T_PR * const AMREX_RESTRICT u2z = soa_2.m_rdata[PIdx::uz];

    // Ionization level of each macroparticle, for the species with field ionization
    int const * const AMREX_RESTRICT ion_lev1 =
        (ionization_level_comp1 >= 0) ? soa_1.m_runtime_idata[ionization_level_comp1] : nullptr;
    int const * const AMREX_RESTRICT ion_lev2 =
        (ionization_level_comp2 >= 0) ? soa_2.m_runtime_idata[ionization_level_comp2] : nullptr;

    // compute Debye length lmdD (if not using a fixed L = Coulomb log)
    T_PR lmdD = T_PR(-1.0);
    if ( L <= T_PR(0.0) ) {
        if (global_lamdb <= T_PR(0.0)) {
            // For a species with field ionization, n*q^2 is replaced by the sum of n_i*(Z_i*q)^2
            // over the charge states, i.e. q^2 is multiplied by the weighted average of Z^2
            T_PR Z2_avg1 = T_PR(1.0);
            T_PR Z2_avg2 = T_PR(1.0);
            if (ion_lev1 != nullptr) {
                T_PR sum_w = T_PR(0.0), sum_wZ2 = T_PR(0.0);
                for (T_index i = I1s; i < I1e; ++i) {
                    T_PR const Z = T_PR(ion_lev1[ I1[i] ]);
                    sum_w += w1[ I1[i] ];
                    sum_wZ2 += w1[ I1[i] ]*Z*Z;
                }
                Z2_avg1 = (sum_w > T_PR(0.0)) ? sum_wZ2/sum_w : T_PR(0.0);
            }
            if (ion_lev2 != nullptr) {
                T_PR sum_w = T_PR(0.0), sum_wZ2 = T_PR(0.0);
                for (T_index i = I2s; i < I2e; ++i) {
                    T_PR const Z = T_PR(ion_lev2[ I2[i] ]);
                    sum_w += w2[ I2[i] ];
                    sum_wZ2 += w2[ I2[i] ]*Z*Z;
                }
                Z2_avg2 = (sum_w > T_PR(0.0)) ? sum_wZ2/sum_w : T_PR(0.0);
            }
            // A species of neutral atoms does not contribute to the screening
            T_PR inv_lmdD2 = T_PR(0.0);
            if (Z2_avg1 > T_PR(0.0)) { inv_lmdD2 += n1*q1*q1*Z2_avg1/(T1*PhysConst::epsilon_0); }
            if (Z2_avg2 > T_PR(0.0)) { inv_lmdD2 += n2*q2*q2*Z2_avg2/(T2*PhysConst::epsilon_0); }
            lmdD = (inv_lmdD2 > T_PR(0.0)) ? T_PR(1.0)/std::sqrt(inv_lmdD2) : T_PR(0.0);
        } else {
            lmdD = global_lamdb;
        }
    }

    // compute atomic spacing
    const T_PR maxn = amrex::max(n1,n2);
    const auto rmin = static_cast<T_PR>( 1.0/std::cbrt(4.0*MathConst::pi/3.0*maxn) );

    // bmax (screening length) cannot be smaller than atomic spacing
    const T_PR bmax = amrex::max(lmdD, rmin);

    // set max cross section based on mfp = atomic spacing
    const T_PR sigma_max = T_PR(1.0) / (maxn * rmin);

    // call UpdateMomentumPerezElastic()
    {
      T_index i1 = I1s + coll_idx;
      T_index i2 = I2s + coll_idx;

      // we will start from collision number = coll_idx and then add
      // stride (smaller set size) until we do all collisions (larger set size)
      for (T_index k = coll_idx; k < max_N; k += min_N)
      {

          // Compute the effective density n12 used to compute the normalized
          // transport mean free path (s12) in UpdateMomentumPerezElastic().
          // s12 is defined such that the expected value of the change in particle
          // velocity is equal to that from the full NxN pairing method for each particle, as described in
          // Justin Ray Angus, Yichen Fu, Vasily Geyko, Dave Grote, David Larson, JCP 531, 113927 (2025).
          // This method is a direct extension of the original binary-pairing method for equally weighted
          // particles by Takizuka and Abe JCP 25 (1977).
          T_PR n12;
          const T_PR wpmax = amrex::max(w1[ I1[i1] ],w2[ I2[i2] ]);
          if (isSameSpecies) { n12 = wpmax*static_cast<T_PR>(min_N+max_N-1)/dV; }
          else { n12 = wpmax*static_cast<T_PR>(min_N)/dV; }

          // Charges of the two macroparticles: for a species with field ionization, the charge
          // is proportional to the ionization level (neutral atoms have a zero cross section)
          T_PR const q1_pair = (ion_lev1 != nullptr) ? q1*T_PR(ion_lev1[ I1[i1] ]) : q1;
          T_PR const q2_pair = (ion_lev2 != nullptr) ? q2*T_PR(ion_lev2[ I2[i2] ]) : q2;

          UpdateMomentumPerezElastic(
              u1x[ I1[i1] ], u1y[ I1[i1] ], u1z[ I1[i1] ],
              u2x[ I2[i2] ], u2y[ I2[i2] ], u2z[ I2[i2] ],
              q1_pair, m1, w1[ I1[i1] ], q2_pair, m2, w2[ I2[i2] ],
              n12, sigma_max, L, bmax, scattering_model, dt, engine );

          if (max_N == NI1) {
            i1 += min_N;
          }
          if (max_N == NI2) {
            i2 += min_N;
          }
      }
    }

}

#endif // WARPX_PARTICLES_COLLISION_ELASTIC_COLLISION_PEREZ_H_
