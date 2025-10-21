"""
reemission.river
~~~~~~~~~~~~~~~~

This module defines a placeholder `River` class representing
a generic river section. The class is part of the evolving
Re-Emission framework but is **not yet integrated** into the
core emission estimation workflow. It is retained here for
future extensions involving riverine carbon fluxes and
reservoir–river interactions.
"""
import warnings
from dataclasses import dataclass
from reemission.interfaces import IRiver


@dataclass
class River(IRiver):
    """ 
    Representation of a generic river section.
    The river can either feed the reservoir or be a reservoir's outlet
    
    ⚠️ **Note:** This class is not currently used in reservoir
    greenhouse gas emission calculations. It remains included
    for future model extensions (e.g., inflow/outflow routing,
    carbon and nutrient transport between reservoirs and rivers).

    Attributes:
        flow (float): Mean annual discharge in m3/s.
        toc (float): Total organic carbon in mg/L (default 0.0 mg/L)
        cod (float): Chemical Oxygen Demand in mgO2/L (default 0.0 mg/L)
        bod5 (float): Biological Oxygen Demand in 5 days of incubation, in mgO2/ (default 0.0 mg/L)
        toc (float): Total organic carbon in mg/L (default 0.0 mg/L)
        tn (float): Total Nitrogen in mg/L (default 0.0 mg/L)
        tp (float): Total Phosphorus in mg/L (default 0.0 mg/L)
        tn (float): Total Suspended Solids in mg/L (default 0.0 mg/L)
        tn (float): Volatile Suspended Solids in mg/L (default 0.0 mg/L)
    """
    flow: float # mean annual discharge in m3/s
    width: float = 0.0
    toc: float = 0.0
    cod: float = 0.0
    bod5: float = 0.0
    tn: float = 0.0
    tp: float = 0.0
    tss: float = 0.0
    vss: float = 0.0
    
    def __post_init__(self):
        warnings.warn(
            "The River class is currently not integrated into emission "
            "calculations. It is a placeholder for future extensions "
            "related to riverine carbon and nutrient fluxes.",
            category=UserWarning,
            stacklevel=2
        )
    
    def area(self, ) -> float:
        """Estimate river cross-sectional area in m2 using an empirical formula.
        
        Returns:
            float: Estimated cross-sectional area in m2.
        """
        # Empirical formula for river cross-sectional area based on flow
        return 0.1 * (self.flow ** 0.8)  # Example formula, can be adjusted
    
if __name__ == "__main__":
    r1 = River(flow=32.56)
