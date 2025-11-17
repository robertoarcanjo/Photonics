# Photonics - Theory and Simulations

A repository dedicated to the implementation and organization of theoretical content related to the study of photonic circuit design in silicon.

# Contents

- Class materials;

- Numerical solver;

# Numerical Solver

## Main sctructure

```mermaid
flowchart LR
    subgraph Input
        direction TB
        param@{ shape: doc, label: "parameters.json" }
    end
    param --> Software
    subgraph Software
        direction TB
        A[Initialize Solver] --> B[Waveguide Geometry]
        B --> C[Grid]
        C --> D[Equations]
        D --> E[Solve]
    end
    F[Analysis]
    subgraph Output
        out@{ shape: doc, label: "Solution.HDF5" }
    end
    Software --> Output
    Output --> F
```

### References

- Orfanidis, Sophocles J. "Electromagnetic waves and antennas." (2008).