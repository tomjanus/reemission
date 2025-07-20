""" """
from dataclasses import dataclass

@dataclass
class River:
    """ 
    Representation of a generic river section.
    The river can either feed the reservoir or be a reservoir's outlet

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
    toc: float = 0.0
    cod: float = 0.0
    bod5: float = 0.0
    tn: float = 0.0
    tp: float = 0.0
    tss: float = 0.0
    vss: float = 0.0
    
if __name__ == "__main__":
    r1 = River(flow=32.56)
