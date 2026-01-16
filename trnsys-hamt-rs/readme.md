# TRNSYS HAMT Component in Rust

This project is a reimplementation of the TRNSYS hygrothermal wall model (Type 395) in Rust. The goal is to create a modern, robust, and well-documented version of the component, based on the physics described in the original Fortran code and the clearer software architecture of the C++ implementation from EnergyPlus (`HeatBalanceHAMTManager`).

The new implementation will maintain strict compatibility with the original's interface, using the same parameters, inputs, and outputs.

---

## 1. Scientific Background

The component models the one-dimensional, non-isothermal transport of heat and moisture through porous building materials. This is a coupled problem, as thermal properties depend on moisture content and moisture transport depends on temperature.

### 1.1. Governing Physics

The model is based on the following principles:
- **Heat Transport:** Conduction through the material matrix and stored water, plus latent heat transport due to phase changes (evaporation/condensation).
- **Moisture Transport:** A combination of vapor diffusion (driven by vapor pressure gradients) and liquid transport (driven by capillary forces and surface tension, represented by suction and redistribution curves).

### 1.2. Numerical Method

A **Forward (Implicit) Finite Difference Method** is used to solve the coupled heat and moisture balance equations. The material is discretized into a series of control volumes or "cells". For each cell and at each time step, an iterative process is used to solve for the temperature and relative humidity that satisfy the conservation laws.

For a detailed breakdown of the governing equations and numerical implementation, please see the [HAMT Calculation Steps Analysis](./doc/calculation_steps.md).

---

## 2. Calculation Workflow

The simulation follows a precise workflow, executed by the TRNSYS engine.

### 2.1. Initialization (at simulation start)

1.  **Load Configuration:** Reads the material properties and construction layout from a dedicated data file. This includes thermal data, but also complex moisture-dependent tables (e.g., sorption isotherms, liquid transport coefficients).
2.  **Create Mesh:** The wall construction is discretized into a 1D grid of cells, with a potentially finer mesh near the surfaces.
3.  **Create Boundary:** Special "virtual" cells are added at the interior and exterior surfaces to handle convective and radiative boundary conditions.
4.  **Link Mesh:** The adjacency of all cells is determined and stored, forming the calculation graph.
5.  **Set Initial State:** The initial temperature and relative humidity of each cell are set based on the material's initial conditions.

### 2.2. Time Step Execution (at each time step)

This is an iterative process that continues until the solution converges.

**Iteration Loop:**

1.  **Update Boundary Conditions:** Set the temperature and humidity of the virtual boundary cells from the component's inputs (e.g., air temperatures, RH, radiation).
2.  **Update Cell Properties:** For each cell, update its moisture-dependent properties (thermal conductivity, vapor resistance, etc.) based on its current temperature and RH from the previous iteration.
3.  **Solve Heat Balance:** Solve the finite difference equation for the cell's **new temperature**, accounting for conduction, latent heat, and stored energy.
4.  **Solve Moisture Balance:** Solve the finite difference equation for the cell's **new relative humidity**, accounting for vapor diffusion, liquid transport, and stored moisture. This calculation uses the new temperature from the step above, creating the coupling.
5.  **Check Convergence:** If the change in temperature and RH for all cells is below a tolerance, exit the loop. Otherwise, repeat. If the maximum number of iterations is reached, exit with a warning.

### 2.3. End of Time Step

1.  **Commit State:** The converged temperature and RH values are set as the final state for the current time step.
2.  **Store State:** The state of all cells is saved in TRNSYS's dynamic storage array for use in the next time step.
3.  **Calculate Outputs:** Final output values (surface temperatures, heat fluxes, etc.) are calculated.

---

## 3. Data Storage

The component requires storing different types of data across various stages of the simulation.

-   **Simulation Constants:** Loaded once at the start. Includes all material properties, construction definitions, and the cell mesh topology.
-   **Time Step Constants:** Updated at the beginning of each time step. Includes all boundary conditions derived from the component's inputs.
-   **Persistent State (between time steps):** The temperature and relative humidity of every cell in the mesh.

For a detailed breakdown, see [Data Storage Requirements](./doc/calculation_steps.md#4-data-storage).

---

## 4. Interface Contract (Type 395)

The interface is identical to the legacy Fortran `Type395_mh.f90`.

**Parameters:**
1.  Number of wall layers (No Actual Use...)
2.  Surface area (m²)
3.  Logical unit (LU) for HAMT data file
4.  *(Unused)*
5.  Depth of the user-defined node for additional heat gain/loss (m)
6.  Flag to enable/disable report file writing
7.  Output interval for report files (hours)
8.  Output file ID string

**Inputs (11 total):**
1.  Outside air temperature (°C)
2.  Inside air temperature (°C)
3.  Outside air relative humidity (%)
4.  Inside air relative humidity (%)
5.  Outside heat transfer coefficient (kJ/hr·m²·K)
6.  Inside heat transfer coefficient (kJ/hr·m²·K)
7.  Shortwave radiation/gain on outside surface (kJ/hr)
8.  Shortwave radiation/gain on inside surface (kJ/hr)
9.  Outside mass transfer coefficient (kg/m²·s)
10. Inside mass transfer coefficient (kg/m²·s)
11. Specific heat gain at user-defined node (kJ/hr)

**Outputs (11 total):**
1.  Exterior surface temperature (°C)
2.  Interior surface temperature (°C)
3.  Exterior surface relative humidity (%)
4.  Interior surface relative humidity (%)
5.  Temperature at user-defined depth (°C)
6.  Relative humidity at user-defined depth (%)
7.  Water content at user-defined depth (kg/m³)
8.  Exterior surface specific heat flux (kJ/hr·m²)
9.  Interior surface specific heat flux (kJ/hr·m²)
10. Exterior surface moisture flux (kg/h)
11. Interior surface moisture flux (kg/h)

---

## 5. Rust Implementation Plan

The project will be structured into several small, well-defined modules under the `src/hamt/` directory.

-   `hamt_manager.rs`: The main orchestrator, holding the simulation state and managing the workflow.
-   `material.rs`: Structs and methods for storing and interpolating material properties.
-   `cell.rs`: Definition of the `SubCell` struct and related data.
-   `io.rs`: Functions for parsing the HAMT data file.
-   `physics.rs`: Contains the core psychrometric and transport equations.

The main `src/lib.rs` will handle the interface with the TRNSYS engine, calling the `hamt_manager` to perform the calculations at each stage of the simulation.
