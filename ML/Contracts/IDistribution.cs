using System;
using System.Collections.Generic;
using ML.Core;
using ML.Core.Distributions;

namespace ML.Contracts
{
  /// <summary>
  /// Marker interface for parameters of distribution
  /// </summary>
  public interface IDistributionParameters
  {
  }

  /// <summary>
  /// Contract for probability distribution (discrete or continuous)
  /// </summary>
  public interface IDistribution<TParam>
    where TParam : IDistributionParameters
  {
    /// <summary>
    /// Parameters of distribution
    /// </summary>
    TParam Params { get; set; }

    /// <summary>
    /// Returns value of probalility (in the case of discrete distribution)
    /// or probability density (in the case of continuous distribution)
    /// </summary>
    double Value(double x);

    /// <summary>
    /// Fills distrubution parameters with Maximum Likelihood estimation
    /// </summary>
    void MaximumLikelihood(double[] sample);

    /// <summary>
    /// Fills distrubution parameters with Maximum Likelihood estimation from a given classified sample.
    /// Ranges result with respect to classes and feature indices
    /// </summary>
    Dictionary<ClassFeatureKey, TParam> MaximumLikelihood(ClassifiedSample<double[]> sample);
  }
}
