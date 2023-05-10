# Main source code of the RE-Emission package

#### profile.py
`profile.py` contains the functionality for creating emission profiles, adding profiles together, plotting individual and composite profiles from multiple reservoirs and calculating total emissions.

Features:
* Suport for storing emission values with units
* Calculation of emission profiles for time > time_horizon (100 years)
* Conversion of emission profiles to EmissionProfileSeries (using Pandas)
* Interpolation of emisison profiles, e.g. to create profiles with equally spaced years, e.g. at 1 year or 2 year time-steps.
* Addition of multiple profiles in `EmissionProfileSeries` format to.
* Plotting combined profiles from multiple reservoirs.

UML Diagram:

```mermaid
classDiagram
    class AssetConstructionStage{
        <<enum>>
        +EXISTING
        +FUTURE
    }
    class AssetSimDateManager{
        +Date construction_date
        +Date sim_start_date
        -year_to_date(year)$
        +asset_status() AssetConstructionStage
        +asset_age() Int
    }
    class EmissionQuantity{
        +Float value
        +String unit
        +convert_unit(new_unit)
    }
    class EmissionProfile{
        <<iterator>>
        +List~EmissionQuantity~ values
        +List~Int~ years
        -_check_units()
        -_is_equally_spaced(spacing Int) Bool
        +convert_unit(new_unit)
        +unit(check_consistency) Bool
        +interpolate(spacing Int) EmissionProfile
        +plot()
        +to_series(construction_year: Int, interpolate: Bool) EmissionProfileSeries
    }
    class EmissionProfileSeries{
        +Series~EmissionQuantity~: values
        +String unit
        +plot()
    }
    class CombinedEmissionProfile{
        -List~EmissionProfileSeries~
        +EmissionProfileSeries profile
        +unit() String
        -_check_units(emission_profiles)$
        -_combine_profiles(emission_profiles)$ EmissionProfileSeries
        +plot()
    }
    class EmissionProfileCalculator{
        +Int time_horizon$
        +Callable profile_fun
        +emission(no_years) EmissionQuantity
        +calculate(years Iterable~int~) EmissionProfile
    }
    class TotalEmissionCalculator{
        +Int time_horizon$
        +Callable integral_fun
        +calculate(start_year, end_year) EmissionQuantity
    }
    EmissionQuantity --* EmissionProfile
    EmissionProfile --> EmissionProfileSeries : produces
    EmissionProfileCalculator --> EmissionProfile : calculates
    CombinedEmissionProfile --> EmissionProfileSeries : uses
    EmissionProfileCalculator --> EmissionQuantity : depends on
    TotalEmissionCalculator --> EmissionQuantity: calculates
    AssetSimDateManager --> AssetConstructionStage
```
