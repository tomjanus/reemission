""" """
from typing import Dict, Any, List, Optional, Literal, Tuple
from functools import partial
from dataclasses import dataclass, field
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch, Rectangle
from scipy.stats import bootstrap
from scipy.stats import norm
from reemission.salib.runners import SobolResults, SobolScenarioResults


def calculate_pr_half_width(variance: float, pred_interval: float = 95) -> float:
    """ Calculate the half-width of the prediction interval.
    Args:
        variance (float): The variance of the model output.
        pred_interval (float): The prediction interval percentage (default is 95).
    Returns:
        float: The half-width of the prediction interval.
    Raises:
        ValueError: If variance is negative or prediction interval is not between 0 and 100
    """
    if variance < 0:
        raise ValueError("Variance must be non-negative.")
    if not (0 < pred_interval < 100):
        raise ValueError("Prediction interval must be between 0 and 100.")
    # Calculate the upper bound of the prediction interval
    upper_bound = ( pred_interval + (100-pred_interval) / 2 ) / 100
    return norm.ppf(upper_bound) * np.sqrt(variance)


# Compute confidence intervals for mean S1 and ST using bootstrapping
def bootstrap_ci(data: np.ndarray, confidence_level: float, n_resamples: int) -> Tuple[np.ndarray, np.ndarray]:
    """ Compute confidence intervals for the mean of each column in the data using bootstrapping.
    Args:
        data (np.ndarray): 2D array of shape (n_samples, n_features).
        confidence_level (float): Confidence level for the intervals (e.g., 0.95).
        n_resamples (int): Number of bootstrap resamples to perform.
    Returns:
        Tuple[np.ndarray, np.ndarray]: Lower and upper bounds of the confidence intervals for each column.
    Raises:
        ValueError: If confidence_level is not between 0 and 1 or n_resamples is not positive.
    """
    alpha = 1 - confidence_level
    lower_bounds = []
    upper_bounds = []
    # Perform bootstrapping for each parameter (column)
    for col in range(data.shape[1]):
        resampled_means = [
            np.mean(np.random.choice(
                data[:, col],
                size=len(data[:, col]),
                replace=True)
            )
            for _ in range(n_resamples)
        ]
        lower_bounds.append(np.percentile(resampled_means, alpha / 2 * 100))
        upper_bounds.append(np.percentile(resampled_means, (1 - alpha / 2) * 100))
    return np.array(lower_bounds), np.array(upper_bounds)

 
@dataclass
class SobolScenarioResultsVisualizer:
    """ Visualizes Sobol sensitivity results with defined scenarios.
    Attributes:
        sc_results (SobolScenarioResults): The Sobol scenario results to visualize.
    """
    sc_results: SobolScenarioResults
    
    def plot_S1_ST(
            self,
            ax: plt.Axes,
            title: str = 'Sobol Sensitivity Indices across scenarios',
            x_label_rotation: float = 45,
            confidence_level: float = 0.95,
            tight_layout: bool = True) -> plt.Axes:
        """Plot Sobol indices S1 and ST.
        Uses bootstrap resampling to compute confidence intervals for the indices.
        This method collects all S1 and ST indices across scenarios, computes their means,
        and calculates confidence intervals using bootstrap resampling.
        It then plots the mean indices with asymmetric error bars representing the confidence intervals.
        The error bars are calculated as the difference between the mean and the lower/upper bounds of
        the confidence intervals. 
        Args:
            ax (plt.Axes): The axes to plot on.
            title (str): The title of the plot.
            x_label_rotation (float): Rotation angle for x-axis labels.
            confidence_level (float): Confidence level for error bars.
            tight_layout (bool): Whether to apply tight layout to the plot.
        Returns:
            plt.Axes: The axes with the plot.
        """
        params = self.sc_results.var_names
        x = np.arange(len(params))
        width = 0.35

        # Collect all S1 and ST vectors into two NumPy 2D arrays: shape (n_simulations, n_parameters)
        all_S1_array = np.array([result.indices['S1'] for result in self.sc_results.results])
        all_ST_array = np.array([result.indices['ST'] for result in self.sc_results.results])
        # Compute the average S1 and ST index across simulations (i.e., mean across rows)
        mean_S1 = np.mean(all_S1_array, axis=0)
        mean_ST = np.mean(all_ST_array, axis=0)
        var_S1 = np.var(all_S1_array, axis=0, ddof=1)
        var_ST = np.var(all_ST_array, axis=0, ddof=1)
        calc_width = partial(calculate_pr_half_width, pred_interval=confidence_level * 100)
        error_S1 = list(map(calc_width, var_S1))
        error_ST = list(map(calc_width, var_ST))

        # Plot bars with asymmetric error bars
        ax.bar(
            x, mean_S1, width, yerr=error_S1, label='First-order', 
            capsize=3, 
            error_kw={'elinewidth': 0.5}
        )
        ax.bar(
            x+width, mean_ST, width, yerr=error_ST, label='Total-order', 
            capsize=3, 
            error_kw={'elinewidth': 0.5}
        )
        ax.set_xticks(x+width/2)
        ax.set_xticklabels(params, rotation=x_label_rotation)
        ax.set_ylabel('Sobol index')
        ax.set_xlabel('Variable')
        ax.set_title(title)
        ax.legend()
        if tight_layout:
            plt.tight_layout()
        return ax

    def plot_variance_per_scenario(
            self,
            ax: plt.Axes,
            title: str = 'Scenario-specific uncertainty decomposition',
            tight_layout: bool = True) -> plt.Axes:
        """ Plot variance contributions for each scenario.
        Args:
            ax (plt.Axes): The axes to plot on.
            title (str): The title of the plot.
            tight_layout (bool): Whether to apply tight layout to the plot.
        Returns:
            plt.Axes: The axes with the plot.
        """
        def results_to_dict() -> List[Dict[str, Any]]:
            outputs = []
            for ix, variance in enumerate(self.sc_results.variances):
                output_data = {}
                output_data['Scenario'] = self.sc_results.scenario_names[ix]
                output_data['Variance'] = variance.total_variance
                for group_name, group_variance in variance.contributions_by_group.items():
                    output_data[group_name] = group_variance
                #print(output_data)
                outputs.append(output_data)
            return outputs
        outputs = results_to_dict()
        df_scenarios = pd.DataFrame(outputs)
        df_melted = df_scenarios.melt(
            id_vars=df_scenarios.columns[:2], 
            value_vars=df_scenarios.columns[2:],
            var_name='Group', value_name='GroupVariance'
        )
        df_melted['Fraction'] = df_melted['GroupVariance'] / df_melted['Variance']
        sns.barplot(ax=ax, data=df_melted, x='Scenario', y='Fraction', hue='Group')
        ax.set_ylabel('Fraction of variance')
        ax.set_title(title)
        ax.legend(title='Parameter Group')
        if tight_layout:
            plt.tight_layout()
        return ax
    
    def plot_outputs_per_scenarios_simple(
            self,
            ax: plt.Axes,
            title: str = 'Outputs across multiple scenarios',
            x_label_rotation: float = 45,
            confidence_level: float = 0.95,
            tight_layout: bool = True,
            sorting: Optional[Literal['asc', 'desc']] = None,
            scenario_names: Optional[List[str]] = None,
            connecting_linestyle: Optional[str] = None) -> plt.Axes:
        """Plot model outputs across scenarios with simple error bars.
        
        Args:
            ax (plt.Axes): The axes to plot on.
            title (str): The title of the plot.
            connecting_linestyle (Optional[str]): Line style for connecting points across scenarios.
                Use None for no connecting lines.
                
        Returns:
            plt.Axes: The axes with the plot.
        """
        if not scenario_names:
            # If no scenario names provided, use the ones from the results
            scenario_names = self.sc_results.scenario_names
        n_scenarios = len(scenario_names)
        x = np.arange(n_scenarios)
        
        # Get outputs and variance for each scenario
        #outputs = [result.nominal_output for result in self.sc_results.results]
        outputs = [np.mean(result.outputs) for result in self.sc_results.results]
        variances = [variance.total_variance for variance in self.sc_results.variances]
        calc_width = partial(calculate_pr_half_width, pred_interval=confidence_level * 100)
        pis = list(map(calc_width, variances))
        
        if sorting:
            # Sort outputs and prediction intervals based on the specified order
            if sorting == 'asc':
                sorted_indices = np.argsort(outputs)
            elif sorting == 'desc':
                sorted_indices = np.argsort(outputs)[::-1]
            else:
                raise ValueError("Invalid sorting option. Use 'asc' or 'desc'.")
            outputs = [outputs[i] for i in sorted_indices]
            pis = [pis[i] for i in sorted_indices]
            scenario_names = [scenario_names[i] for i in sorted_indices]
        
        # Add connecting lines between scenarios if requested
        if connecting_linestyle:
            ax.plot(x, outputs, linestyle=connecting_linestyle, 
                    linewidth=1, color='grey', zorder=1)
        
        # Plot scatter points with error bars
        ax.errorbar(x, outputs, yerr=pis, fmt='o', 
                    ecolor='black', capsize=4, capthick=1, 
                    elinewidth=0.8, markersize=6, markerfacecolor='white',
                    markeredgecolor='black', markeredgewidth=0.8,
                    label='Model output', zorder=10)
        
        # Set labels and styling
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_names, rotation=x_label_rotation)
        ax.set_ylabel('Model output')
        ax.set_xlabel('Scenario')
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        if tight_layout:
            plt.tight_layout()
        return ax
    
    def plot_outputs_per_scenarios(
            self,
            ax: plt.Axes,
            title: str = 'Outputs across multiple scenarios',
            x_label_rotation: float = 45,
            confidence_level: float = 0.95,
            width: float = 0.3,
            scenario_names: Optional[List[str]] = None,
            connecting_linestyle: Optional[str] = None,
            sorting: Optional[Literal['asc', 'desc']] = None,
            component_colors: Optional[List] = None,
            tight_layout: bool = True,
            x_label: str = "Reservoir") -> plt.Axes:
        """ Plot model outputs across scenarios with stacked error bars.
        Args:
            ax (plt.Axes): The axes to plot on.
            title (str): The title of the plot.
            x_label_rotation (float): Rotation angle for x-axis labels.
            confidence_level (float): Confidence level for the prediction intervals.
            width (float): Width of the bars in the plot.
            scenario_names (Optional[List[str]]): Names of the scenarios to display on the x-axis.
            connecting_linestyle (Optional[str]): Line style for connecting points across scenarios.
            sorting (Optional[Literal['asc', 'desc']]): Sorting order for the outputs.
            component_colors (Optional[List]): Colors for the components in the stacked bars.
            tight_layout (bool): Whether to apply tight layout to the plot.
            x_label (str): Label for the x-axis.
        Returns:
            plt.Axes: The axes with the plot.
        """
        if not scenario_names:
            # If no scenario names provided, use the ones from the results
            scenario_names = self.sc_results.scenario_names
        n_scenarios = len(scenario_names)
        group_names = self.sc_results.group_names
        n_groups = len(group_names)
        x = np.arange(n_scenarios)
        outputs = [np.mean(result.outputs) for result in self.sc_results.results]
        sobol_variances = [variance for variance in self.sc_results.variances]
        
        if sorting:
            # Sort outputs and prediction intervals based on the specified order
            if sorting == 'asc':
                sorted_indices = np.argsort(outputs)
            elif sorting == 'desc':
                sorted_indices = np.argsort(outputs)[::-1]
            else:
                raise ValueError("Invalid sorting option. Use 'asc' or 'desc'.")
            outputs = [outputs[i] for i in sorted_indices]
            sobol_variances = [sobol_variances[i] for i in sorted_indices]
            #pis = [pis[i] for i in sorted_indices]
            scenario_names = [scenario_names[i] for i in sorted_indices]     
                
        variance_components = []
        pis_values = []
        for i in range(len(sobol_variances)):
            variances = list(sobol_variances[i].contributions_by_group.values())
            calc_width = partial(calculate_pr_half_width, pred_interval=confidence_level * 100)
            pis = list(map(calc_width, variances))
            variance_components.extend([variances])
            pis_values.extend([pis])
        variance_components = np.array(variance_components)
        pis_components = np.array(pis_values)
                    
        if connecting_linestyle:
            ax.plot(x, outputs, linestyle=connecting_linestyle, linewidth=0.75, color='grey', zorder=2)
        # Plot model outputs
        ax.scatter(
            x, outputs, color='white', zorder=30, 
            label='Model output', marker='s', s=30, edgecolor='black', linewidths=0.7)
        # Use a perceptually uniform colormap (or 'tab10', 'tab20' for categorical)
        cmap = cm.get_cmap('tab10') # You can try 'Set2', 'tab20', 'viridis', etc.
        # Generate evenly spaced colors in the colormap
        if not component_colors:
            component_colors = [cmap(i / n_groups) for i in range(n_groups)]
        
        cap_width, bar_width = width, width
        line_width = 6.5
        cap_thickness = 1
        
        for i in range(n_scenarios):
            y = outputs[i]
            sc_name = self.sc_results.sc_names[i]
            #err = calc_width(sobol_variances[i].total_variance)
            err = calc_width(sum(variance_components[i, :]))
            contributions = sobol_variances[i].contributions_by_group            
            
            #ax.hlines([y - err, y + err], x[i] - cap_width, x[i] + cap_width,
            #        color='k', linewidth=cap_thickness, zorder=2)
            
            # --- Symmetric stacked decomposition ---
            base_top = y
            base_bottom = y
            for j in range(pis_components.shape[1]):
                comp_height = pis_components[i, j]  # Full contribution in each direction
                # Top segment (upward)
                top = base_top + comp_height
                ax.vlines(x[i], base_top, top,
                        color=component_colors[j], 
                        linewidth=line_width, 
                        alpha=0.9, 
                        zorder=-4)
                # Rectangle contour (top)
                rect_top = Rectangle(
                    (x[i] - bar_width / 2, base_top),
                    width=bar_width,
                    height=comp_height,
                    edgecolor='black',
                    facecolor='none',
                    linewidth=0.5,
                    zorder=5
                )
                ax.add_patch(rect_top)
                base_top = top
                # Bottom segment
                bottom = base_bottom - comp_height
                ax.vlines(x[i], base_bottom, bottom,
                        color=component_colors[j], 
                        linewidth=line_width, 
                        alpha=0.9, 
                        zorder=-4)
                # Rectangle contour (bottom)
                rect_bottom = Rectangle(
                    (x[i] - bar_width / 2, bottom),
                    width=bar_width,
                    height=comp_height,
                    edgecolor='black',
                    facecolor='none',
                    linewidth=0.5,
                    zorder=5
                )
                ax.add_patch(rect_bottom)
                base_bottom = bottom
            
        # Labels and layout
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_names, rotation=x_label_rotation)
        ax.set_ylabel('Model output')
        ax.set_xlabel(x_label)
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.7)
        legend_elements = [
            Patch(facecolor=c, label=l) for c, l in zip(component_colors, group_names)
        ]
        legend_elements.append(
            plt.Line2D(
                [0], [0], marker='s', color='w', label='Model output',
                markerfacecolor='white', markersize=6, markeredgecolor='black')
        )
        ax.legend(handles=legend_elements, loc='upper right')
        if tight_layout:
            plt.tight_layout()
        return ax
    
    def plot_variance_contributions_by_group(
            self,
            ax: plt.Axes,
            title: str = 'Variance contribution by uncertainty group',
            confidence_level: float = 0.95,
            tight_layout: bool = True,
            plot_legend: bool = True) -> plt.Axes:
        """Plots variance contribution for each parameter uncertainty group across all scenarios.
        Args:
            ax (plt.Axes): The axes to plot on.
            title (str): The title of the plot.
            confidence_level (float): Confidence level for the prediction intervals.
            tight_layout (bool): Whether to apply tight layout to the plot.
            plot_legend (bool): Whether to show the legend.
        Returns:
            plt.Axes: The axes with the plot.
        """
        # Check if groups are defined and consistent across all results
        all_groups = []
        for i, result in enumerate(self.sc_results.results):
            if hasattr(result, 'groups') and result.groups:
                all_groups.append((i, result.groups))
            else:
                print(
                    f"Result {i} has no groups defined\n"
                    "- this plot requires groups to be defined for each result\n"
                    "and be consistent across all results"
                )
                return ax
                
        # Check if all groups are the same
        if len(all_groups) > 1:
            is_consistent = all(
                all_groups[0][1] == groups for _, groups in all_groups[1:])
            if is_consistent:
                print(
                    f"All {len(all_groups)} results have the same groups: {all_groups[0][1]}")
                labels = all_groups[0][1]
                x = np.arange(len(labels))
                width = 0.35
            else:
                print("Groups are inconsistent between results:")
                for i, groups in all_groups:
                    print(f"Result {i}: {groups}")
                return ax
            
        # Initialize lists to store S1 and ST values for each group across all scenarios
        all_S1_by_group = []
        all_ST_by_group = []        
            
        for result in self.sc_results.results:
            # Get S1 values for each group (or 0 if not present)
            s1_values = [result.grouped_indices.get('S1', {}).get(group, 0) for group in labels]
            # Get ST values for each group (or 0 if not present)
            st_values = [result.grouped_indices.get('ST', {}).get(group, 0) for group in labels]
            
            all_S1_by_group.append(s1_values)
            all_ST_by_group.append(st_values)
        
        # Convert to numpy arrays for easier manipulation
        all_S1_by_group = np.array(all_S1_by_group)
        all_ST_by_group = np.array(all_ST_by_group)
        
        # Calculate mean and standard error for each group
        mean_S1_by_group = np.mean(all_S1_by_group, axis=0)
        mean_ST_by_group = np.mean(all_ST_by_group, axis=0)
        
        # Calculate confidence interval, if we have more than one scenario
        calc_width = partial(calculate_pr_half_width, pred_interval=confidence_level * 100)
        if all_S1_by_group.shape[0] > 1:
            var_S1 = np.var(all_S1_by_group, axis=0, ddof=1)
            var_ST = np.var(all_ST_by_group, axis=0, ddof=1)
            error_S1 = list(map(calc_width, var_S1))
            error_ST = list(map(calc_width, var_ST))
            # Plot bars with error bars
            ax.bar(x - width/2, mean_S1_by_group, width, yerr=error_S1, 
                label='First-order', capsize=3, error_kw={'elinewidth': 0.5})
            ax.bar(x + width/2, mean_ST_by_group, width, yerr=error_ST, 
                label='Total-order', capsize=3, error_kw={'elinewidth': 0.5})
        else:
            # If only one scenario, no error bars needed
            ax.bar(x - width/2, mean_S1_by_group, width, label='First-order')
            ax.bar(x + width/2, mean_ST_by_group, width, label='Total-order')
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20)
        ax.set_ylabel('Sum of Sobol indices')
        ax.set_title(title)
        if plot_legend:
            ax.legend()
        if tight_layout:
            plt.tight_layout()
        return ax


@dataclass
class SobolResultVisualizer:
    """ Visualizes Sobol sensitivity analysis results.
    Attributes:
        results (SobolResults): The Sobol results to visualize.
    """
    results: SobolResults
    par_name_map: Dict[str, str] = field(default_factory=dict)
    
    def plot_S1_ST(
            self,
            ax: plt.Axes,
            title: str = 'Sobol Sensitivity Indices',
            x_label_rotation: float = 45,
            tight_layout: bool = True) -> plt.Axes:
        """Plot Sobol indices S1 and ST. 
        Args:
            ax (plt.Axes): The axes to plot on.
            title (str): The title of the plot.
            tight_layout (bool): Whether to apply tight layout to the plot.
        Returns:
            plt.Axes: The axes with the plot.
        """
        params = self.results.var_names
        # Translate param names if param in par_name_map
        if self.par_name_map:
            params = [self.par_name_map.get(p, p) for p in params]
        x = np.arange(len(params))
        width = 0.35
        ax.bar(x, self.results.indices['S1'], width, label='First-order')
        ax.bar(x+width, self.results.indices['ST'], width, label='Total-order')
        ax.set_xticks(x+width/2)
        ax.set_xticklabels(params, rotation=x_label_rotation)
        ax.set_ylabel('Sobol index')
        ax.set_xlabel('Variable')
        ax.set_title(title)
        ax.legend()
        if tight_layout:
            plt.tight_layout()
        return ax

    def plot_variance_contribution_by_group(
            self,
            ax: plt.Axes,
            title: str = 'Variance contribution by uncertainty group',
            tight_layout: bool = True,
            plot_legend: bool = True) -> plt.Axes:
        """Plots variance contribution for each parameter uncertainty group.
        Args:
            ax (plt.Axes): The axes to plot on.
            title (str): The title of the plot.
            tight_layout (bool): Whether to apply tight layout to the plot.
            plot_legend (bool): Whether to show the legend.
        Returns:
            plt.Axes: The axes with the plot.
        """
        # Only plot if groups are defined
        # TODO: Normalize the indices - should they sum up to 1??? Check what GPT has to say.
        if not self.results.groups:
            return ax
        labels = self.results.groups
        x = np.arange(len(labels))
        width = 0.35
        ax.bar(
            x - width/2, [self.results.grouped_indices['S1'][g] for g in labels],
            width, label='First-order')
        ax.bar(
            x + width/2, [self.results.grouped_indices['ST'][g] for g in labels],
            width, label='Total-order')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20)
        ax.set_ylabel('Sum of Sobol indices')
        ax.set_title(title)
        if plot_legend:
            ax.legend()
        if tight_layout:
            plt.tight_layout()
        return ax

    def plot_output_histogram(
            self, 
            ax: plt.Axes,
            title: str = 'Distribution of model output',
            tight_layout: bool = True,
            plot_legend: bool = True) -> plt.Axes:
        """Plot histogram of model outputs.
        Args:
            ax (plt.Axes): The axes to plot on.
            title (str): The title of the plot.
            tight_layout (bool): Whether to apply tight layout to the plot.
            plot_legend (bool): Whether to show the legend.
        Returns:
            plt.Axes: The axes with the plot.
        """
        nominal_output = self.results.nominal_output
        ax.hist(self.results.outputs, bins=50, density=True, alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel('Y')
        ax.set_ylabel('Density')
        ax.axvline(nominal_output, color='red', linestyle='--', linewidth=2,
              label=f'Nominal (Y={nominal_output:.2f})')
        if plot_legend:
            ax.legend()
        if tight_layout:
            plt.tight_layout()
        return ax

    def plot_output_kde(
            self, 
            ax: plt.Axes,
            confidence_level: float = 0.95,
            ci_type: Optional[Literal['bca', 'percentile']] = None,
            title: str = 'Distribution of model output (Y)',
            tight_layout: bool = True,
            plot_legend: bool = True,
            xlims: Optional[Tuple[float, float]] = None) -> plt.Axes:
        """Improved version of the output histogram plot.
        Args:
            ax (plt.Axes): The axes to plot on.
            confidence_level (float): Confidence level for the intervals.
            ci_type (Optional[Literal['bca', 'percentile']]): Type of confidence interval to use.
            title (str): The title of the plot.
            tight_layout (bool): Whether to apply tight layout to the plot.
            plot_legend (bool): Whether to show the legend.
        Returns:
            plt.Axes: The axes with the plot.
        """
        # Descriptive statistics
        y_sim = self.results.outputs
        y_mean = np.mean(y_sim)
        y_std = np.std(y_sim)
        y_median = np.median(y_sim)
        # Confidence intervals
        alpha = 1 - confidence_level
        nominal_output = self.results.nominal_output
        ci_lower = np.percentile(y_sim, alpha * 100)
        ci_upper = np.percentile(y_sim, confidence_level * 100)
        if ci_type == 'bca':
            # BCa intervals using scipy.stats.bootstrap
            res_mean = bootstrap(
                (y_sim,),
                np.mean,
                confidence_level=confidence_level,
                method='BCa',
                vectorized=False,
                n_resamples=10_000)
            res_median = bootstrap(
                (y_sim,),
                np.median,
                confidence_level=confidence_level, 
                method='BCa',
                vectorized=False, 
                n_resamples=10_000)
            mean_ci = res_mean.confidence_interval
            median_ci = res_median.confidence_interval
        elif ci_type == 'percentile':
            # Percentile-based intervals
            mean_ci = np.percentile(
                [np.mean(np.random.choice(y_sim, size=len(y_sim), replace=True)) for _ in range(1000)],
                [alpha * 100, confidence_level * 100])
            median_ci = np.percentile(
                [np.median(np.random.choice(y_sim, size=len(y_sim), replace=True)) for _ in range(1000)],
                [alpha * 100, confidence_level * 100])
        else:
            pass
        # KDE Plot
        sns.kdeplot(y_sim, ax=ax, fill=True, color='lightgreen', linewidth=2, alpha=0.6)
        # Highlight 95% prediction interval
        ax.axvline(
            y_mean, color='red', linestyle='dashed', linewidth=1.5,
            label=f'Mean = {y_mean:.2f}')
        ax.axvline(
            y_median, color='blue', linestyle='dashed', linewidth=1.5,
            label=f'Median = {y_median:.2f}')
        ax.axvline(nominal_output, color='green', linestyle='dashed', linewidth=1.5,
              label=f'Nominal = {nominal_output:.2f})')
        ax.axvspan(
            ci_lower, ci_upper, color='grey', alpha=0.2)
        ax.axvline(
            ci_lower, color='black', linestyle='-', linewidth=0.5,
            label=f'{alpha * 100:.1f}th Percentile = {ci_lower:.2f}')
        ax.axvline(
            ci_upper, color='black', linestyle='-', linewidth=0.5,
            label=f'{confidence_level * 100:.1f}th Percentile = {ci_upper:.2f}')
        if ci_type in ('bca', 'percentile'):
            ax.axvline(
                mean_ci[0], color='blue', linestyle='dashdot',
                linewidth=1.5, label='Mean CI Lower')
            ax.axvline(
                mean_ci[1], color='blue', linestyle='dashdot',
                linewidth=1.5, label='Mean CI Upper')
            ax.axvline(
                median_ci[0], color='orange', linestyle='dashdot',
                linewidth=1.5, label='Median CI Lower')
            ax.axvline(
                median_ci[1], color='orange', linestyle='dashdot',
                linewidth=1.5, label='Median CI Upper')
        ax.set_title(title)
        ax.set_xlabel('Model Output')
        ax.set_ylabel('Estimated Density')
        if xlims:
            ax.set_xlim(*xlims)
        if plot_legend:
            ax.legend()
        ax.grid(True)
        if tight_layout:
            plt.tight_layout()
        return ax


if __name__ == "__main__":
    pass
