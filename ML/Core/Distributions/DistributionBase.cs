using System;
using System.Collections.Generic;
using ML.Contracts;

namespace ML.Core.Distributions
{
  /// <summary>
  /// Base class for probability distribution (discrete or continuous)
  /// </summary>
  public abstract class DistributionBase : IDistribution
  {
    protected DistributionBase()
    {
    }


    /// <summary>
    /// Returns value of probalility (in the case of discrete distribution)
    /// or probability density (in the case of continuous distribution)
    /// </summary>
    public abstract double Value(double x);

    /// <summary>
    /// Fills distrubution parameters with Maximum Likelihood estimation
    /// </summary>
    public abstract void MaximumLikelihood(double[] sample);

    /// <summary>
    /// Fills distrubution parameters with Maximum Likelihood estimation from a given classified sample.
    /// Ranges result with respect to classes and feature indices
    /// </summary>
    public abstract Dictionary<ClassFeatureKey, IDistribution> MaximumLikelihood(ClassifiedSample<double[]> sample);
  }
}
