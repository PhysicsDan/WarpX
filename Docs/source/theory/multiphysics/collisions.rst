.. _multiphysics-collisions:

Collisions
==========

WarpX includes several different models to capture collisional processes
including collisions between kinetic particles (Coulomb collisions, DSMC,
nuclear fusion, electron-impact ionization) as well as collisions between kinetic particles and a fixed
(i.e. non-evolving) background species (MCC, background stopping).

.. _multiphysics-collisions-mcc:

Background Monte Carlo Collisions (MCC)
---------------------------------------

Several types of collisions between simulation particles and a neutral
background gas are supported including elastic scattering, back scattering,
charge exchange, excitation collisions and impact ionization.

The so-called null collision strategy is used in order to minimize the
computational burden of the MCC module. This strategy is standard in PIC-MCC and
a detailed description can be found elsewhere, for example in :cite:t:`b-Birdsall1991`.
In short the maximum collision probability is found over a sensible range of
energies and is used to pre-select the appropriate number of macroparticles for
collision consideration. Only these pre-selected particles are then individually
considered for a collision based on their energy and the cross-sections of all
the different collisional processes included.

The MCC implementation assumes that the background neutral particles are **thermal**,
and are moving at non-relativistic velocities in the lab frame. For each
simulation particle considered for a collision, a velocity vector for a neutral
particle is randomly chosen given the user specified neutral temperature. The
particle velocity is then boosted to the stationary frame of the neutral through
a Galilean transformation. The energy of the collision is calculated using the
particle utility function, ``ParticleUtils::getCollisionEnergy()``, as

    .. math::

       \begin{aligned}
        E_{coll} &= \sqrt{(\gamma mc^2 + Mc^2)^2 - (mu)^2} - (mc^2 + Mc^2) \\
                 &= \frac{2Mmu^2}{M + m + \sqrt{M^2+m^2+2\gamma mM}}\frac{1}{\gamma + 1}
       \end{aligned}

where :math:`u` is the speed of the particle as tracked in WarpX (i.e.
:math:`u = \gamma v` with :math:`v` the particle speed), while :math:`m` and
:math:`M` are the rest masses of the simulation and background species,
respectively. The Lorentz factor is defined in the usual way,
:math:`\gamma \equiv \sqrt{1 + u^2/c^2}`. Note that if :math:`\gamma\to1` the above
expression reduces to the classical equation
:math:`E_{coll} = \frac{1}{2}\frac{Mm}{M+m} u^2`. The collision cross-sections
for all scattering processes are evaluated at the energy as calculated above.

Once a particle is selected for a specific collision process, that process determines how the particle is scattered as outlined below.

.. _multiphysics-collisions-pulseddecay:

Pulsed Decay
------------

This collision module can be used to have a parent species decay into two product
species with a user-defined decay rate. Mathematically, it solves the following
rate equations on a particle basis:

    .. math::

       \begin{aligned}
        \frac{dN_1}{dt} &= -\nu(t)N_1, \\
        \frac{dN_A}{dt} &= +\nu(t)N_1 = \frac{dN_B}{dt}, \\
       \end{aligned}

where :math:`N_1` is the number of the parent species particles, :math:`N_A` and :math:`N_B` are the numbers of the product species, and :math:`\nu(x,y,z,t)` is the user-specified decay rate.
For each parent particle, the decay probability is calculated as :math:`1-\exp[-\nu(x,y,z,t)\Delta t]`, where :math:`\Delta t` is the time-step size.
If that probability is greater than a randomly chosen number, the parent particle decays into the products whose number and weight is scaled by the integer weight ratio.
After decaying, the parent particle is removed.

This can be used, for example, to represent ionization of a parent species by an externally applied laser pulse.

.. _multiphysics-collisions-dsmc:

Direct Simulation Monte Carlo (DSMC)
------------------------------------

The algorithm by which binary collisions are treated is outlined below. The
description assumes collisions between different species.

1. Particles from both species are sorted by grid-cells.
2. The order of the particles in each cell is shuffled.
3. Within each cell, particles are paired to form collision partners. Particles
   of the species with fewer members in a given cell is split in half so that
   each particle has exactly one partner of the other species.
4. Each collision pair is considered for a collision using the same logic as in
   the MCC description above.
5. Particles that are chosen for collision are scattered according to the
   selected collision process.

Scattering processes
--------------------

Charge exchange
^^^^^^^^^^^^^^^

This process can occur when an ion and neutral (of the same species) collide
and results in the exchange of an electron. The ion velocity is simply replaced
with the neutral velocity and vice-versa.

Elastic scattering
^^^^^^^^^^^^^^^^^^

The ``elastic`` option uses isotropic scattering, i.e., with a differential
cross section that is independent of angle.
This scattering process as well as the ones below that relate to it, are all
performed in the center-of-momentum (COM) frame. Designating the COM velocity of
the particle as :math:`\boldsymbol{u}_c` and its labframe velocity as :math:`\boldsymbol{u}_l`,
the transformation from lab frame to COM frame is done with a general Lorentz
boost (see function ``ParticleUtils::doLorentzTransform()``):

    .. math::
            \begin{bmatrix}
                \gamma_c c \\
                u_{cx} \\
                u_{cy} \\
                u_{cz}
            \end{bmatrix}
         = \begin{bmatrix}
                \gamma & -\gamma\beta_x & -\gamma\beta_y & -\gamma\beta_z \\
                -\gamma\beta_x & 1+(\gamma-1)\frac{\beta_x^2}{\beta^2} & (\gamma-1)\frac{\beta_x\beta_y}{\beta^2} & (\gamma-1)\frac{\beta_x\beta_z}{\beta^2} \\
                -\gamma\beta_y & (\gamma-1)\frac{\beta_x\beta_y}{\beta^2} & 1 +(\gamma-1)\frac{\beta_y^2}{\beta^2} & (\gamma-1)\frac{\beta_y\beta_z}{\beta^2} \\
                -\gamma\beta_z & (\gamma-1)\frac{\beta_x\beta_z}{\beta^2} & (\gamma-1)\frac{\beta_y\beta_z}{\beta^2} & 1+(\gamma-1)\frac{\beta_z^2}{\beta^2} \\
            \end{bmatrix} \begin{bmatrix}
                \gamma_l c \\
                u_{lx} \\
                u_{ly} \\
                u_{lz}
            \end{bmatrix}

where :math:`\gamma` is the Lorentz factor of the relative speed between the lab frame and the COM frame, :math:`\beta_i = v^{COM}_i/c` is the i'th component of the relative velocity between the lab frame and the COM frame with

    .. math::

        \boldsymbol{v}^{COM} = \frac{m \boldsymbol{u}_c}{\gamma_u m + M}

The particle velocity in the COM frame is then isotropically scattered using the function ``ParticleUtils::RandomizeVelocity()``. After the direction of the velocity vector has been appropriately changed, it is transformed back to the lab frame with the reversed Lorentz transform as was done above followed by the reverse Galilean transformation using the starting neutral velocity.

Back scattering
^^^^^^^^^^^^^^^

The process is the same as for elastic scattering above except the scattering angle is fixed at :math:`\pi`, meaning the particle velocity in the COM frame is updated to :math:`-\boldsymbol{u}_c`.

Excitation
^^^^^^^^^^

The process is also the same as for elastic scattering except the excitation energy cost is subtracted from the particle energy. This is done by reducing the velocity before a scattering angle is chosen.

Forward scattering
^^^^^^^^^^^^^^^^^^

This process operates in two ways:

1. If an excitation energy cost is provided, the energy cost is subtracted from the particle energy and no scattering is performed.
2. If an excitation energy cost is not provided, the particle is not scattered and the velocity is unchanged (corresponding to a scattering angle of :math:`0` in the elastic scattering process above).

See :cite:t:`b-Janssen2016` for a recommended use of this process.

Benchmarks
----------

See the :ref:`MCC example <examples-capacitive-discharge>` for a benchmark of the MCC
implementation against literature results.

Particle cooling due to elastic collisions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is straight forward to determine the energy a projectile loses during an elastic collision with another body, as a function of scattering angle, through energy and momentum conservation.
See for example :cite:t:`b-Lim2007` for a derivation. The result is that given a projectile with mass :math:`m`, a target with mass :math:`M`, a scattering angle :math:`\theta`, and collision energy :math:`E`, the post collision energy of the projectile is given by

    .. math::

       \begin{aligned}
       E_{final} = E - &[(E + mc^2)\sin^2\theta + Mc^2 - \cos(\theta)\sqrt{M^2c^4 - m^2c^4\sin^2\theta}] \\
       &\times \frac{E(E+2mc^2)}{(E+mc^2+Mc^2)^2 - E(E+2mc^2)\cos^2\theta}
       \end{aligned}

The impact of incorporating relativistic effects in the MCC routine can be seen in the plots below where high energy collisions are considered with both a classical and relativistic implementation of MCC. It is observed that the classical version of MCC reproduces the classical limit of the above equation but especially for ions, this result differs substantially from the fully relativistic result.

.. figure:: https://user-images.githubusercontent.com/40245517/170900079-74e505a5-2790-44f5-ac84-5847eda954e6.png
   :alt: Comparison of classical and relativistic MCC collision results
   :width: 96%

   Classical v. relativistic MCC.

.. _multiphysics-collisions-electron-impact-ionization:

Electron-impact ionization
--------------------------

The ``electron_impact_ionization`` collision type models the ionization of ions by collisions with free electrons,
:math:`e^- + A^{q+} \rightarrow 2e^- + A^{(q+1)+}`.
This is the dominant ionization mechanism in dense, collisional plasmas in which the electric field is too weak
for field ionization, for instance in solid targets heated by the pedestal of a high-intensity laser pulse.
The ions are represented by a species with field ionization enabled, which provides the ionization level
of each macroparticle and the physical element of the ions.
Field ionization and electron-impact ionization can therefore be used together.

Cross section
^^^^^^^^^^^^^

The ionization cross section of an ion in a given charge state is computed with the empirical formula of
:cite:t:`b-LotzZPhys1967` (see also :cite:t:`b-LotzZPhys1968`), summed over the occupied subshells of the ion:

    .. math::

        \sigma(E) = a \sum_{s,\, P_s < E} q_s \frac{\ln(E/P_s)}{E P_s}

where :math:`a = 4.5 \times 10^{-14}\,\mathrm{cm^2\,eV^2}`, :math:`E` is the kinetic energy of the incident
electron in the rest frame of the ion, and :math:`q_s` and :math:`P_s` are the number of electrons and the
binding energy of the subshell :math:`s`.
The ground-state configuration of each charge state is obtained by filling the subshells in the order
given by the Madelung rule, and the binding energy of a subshell is approximated by the ionization energy
of the charge state in which this subshell is the outermost occupied subshell
(using the same table of ionization energies as for field ionization).
The formula of Lotz is non-relativistic. It is extended to relativistic electrons in the same way as in the
relativistic binary-encounter-Bethe model of :cite:t:`b-KimPRA2000`:

    .. math::

        \sigma(E) = a \sum_{s,\, P_s < E} q_s \frac{\ln(E/P_s) + \left(1 - P_s^2/E^2\right)\delta}{P_s\, m_e c^2 \beta^2/2},
        \qquad \delta = \ln\left(\frac{\beta^2\gamma^2 m_e c^2}{2E}\right) - \beta^2 = \ln\left(\frac{\gamma+1}{2}\right) - \beta^2

where :math:`\gamma` and :math:`\beta` are the Lorentz factor and the normalized velocity of the incident
electron in the rest frame of the ion.
At high energy, the logarithmic term tends to :math:`\ln\left(\beta^2\gamma^2 m_e c^2/2P_s\right) - \beta^2`,
which gives the relativistic rise of the Bethe formula (see also :cite:t:`b-PerezPOP2012`),
while the threshold factor :math:`1 - P_s^2/E^2` ensures that the cross section remains positive above the
ionization threshold of the inner subshells of heavy ions (whose binding energy can be a significant fraction of
:math:`m_e c^2`). The original formula of Lotz is recovered when :math:`E \ll m_e c^2`.

Algorithm
^^^^^^^^^

The electron and ion macroparticles are sorted by cell and paired as in the other binary collision models.
For each pair, the probability of ionization is computed with the algorithm of :cite:t:`b-HigginsonJCP2019`,
which supports macroparticles with arbitrary weights :math:`w_e` and :math:`w_i`:

    .. math::

        P = 1 - \exp\left( -N_{\min}\, w_{\max}\, \sigma v_{rel}\, \Delta t / \Delta V \right)

where :math:`N_{\min}` is the number of macroparticles of the species with fewer macroparticles in the cell,
:math:`w_{\max} = \max(w_e, w_i)`, :math:`\Delta t` is the collision time step and :math:`\Delta V` is the
volume of the cell. As for nuclear fusion, the reaction rate :math:`\sigma v_{rel}` is evaluated in the
center-of-momentum frame of the pair and converted to the lab frame.
If an ionization event occurs, the weight :math:`w_{\min} = \min(w_e, w_i)` is subtracted from both colliding
macroparticles, and three macroparticles of weight :math:`w_{\min}` are created: the scattered electron,
at the position of the incident electron and in the same species, the ion with its ionization level increased
by one, in the same species, and the ejected electron, in the product species.
The ion and the ejected electron are created at the position of the ion, so that the charge is conserved locally.
When the electron and ion macroparticles have different weights, the number of macroparticles thus increases
with each ionization event, and particle resampling may be needed in long simulations.
The accuracy of this algorithm requires the probability :math:`P` to be small for all pairs, which also ensures
that the fact that each ion can be ionized at most once per collision time step does not limit the ionization rate.
Note that, when the electron macroparticles have different weights (which is typically the case after a number of
ionization events, since the scattered electrons have the weight of the ion macroparticles), the probability
:math:`P` of the pairs that involve the electron macroparticles with the largest weights is larger than the mean
ionization probability of the ions per time step, by up to the ratio of these weights to the mean weight of the
electron macroparticles. If :math:`P` is not small for these pairs, the ionization rate is underestimated.
If needed, the collision time step can be reduced with :pp:param:`<collision_name>.ndt_subcycle`.

The kinematics of the ionization events are computed in the rest frame of the ion.
The incident electron loses the ionization energy :math:`I` of the ion plus the kinetic energy :math:`\epsilon`
of the ejected electron.
With the default ``binary_encounter`` model (see :pp:param:`<collision_name>.ejected_electron_energy_model`),
the energy transfer :math:`W = \epsilon + I` is sampled from the binary-encounter (Thomson) distribution
:math:`dP/dW \propto 1/W^2`, for :math:`I \leq W \leq (E + I)/2` (since, by convention, the ejected electron is
the slower of the two outgoing electrons).
With the ``at_rest`` model, :math:`\epsilon = 0`.
The ejected electron is emitted isotropically, the scattered electron keeps the direction of the incident
electron, and the ion recoils such that the momentum is conserved.
The recoil energy of the ion is taken from the scattered electron, such that the total kinetic energy decreases
exactly by :math:`I`.
Note that, as for field ionization, the mass of the ion species does not change when the ions are ionized.
As a consequence, the energy and the momentum are conserved exactly in the rest frame of the ion,
while in the lab frame the ejected electron also carries the kinetic energy and the momentum that it had
when it was bound to the moving ion (and which are not removed from the ion).
These are smaller than the kinetic energy and the momentum of the ion by a factor of order :math:`m_e/m_{ion}`.
Finally, ionization from inner subshells is treated in the same way as ionization from the outermost subshell,
i.e., the energy cost of the event is always the ionization energy of the ion
(the subsequent relaxation of the ion, e.g. by Auger emission, is not modeled).

.. bibliography::
    :keyprefix: b-
