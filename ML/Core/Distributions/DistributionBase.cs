using System;
using System.Collections.Generic;
using ML.Contracts;

namespace ML.Core.Distributions
{
  /// <summary>
  /// Base class for probability 1D distribution (discrete or continuous)
  /// </summary>
  public abstract class DistributionBase<TParam> : IDistribution<TParam>
    where TParam : IDistributionParameters
  {
    protected DistributionBase()
    {
    }

    /// <summary>
    /// Parameters of distribution
    /// </summary>
    public TParam Params { get; set; }

    /// <summary>
    /// Returns value of probalility (in the case of discrete distribution)
    /// or probability density (in the case of continuous distribution)
    /// </summary>
    public abstract double Value(double x);

    /// <summary>
    /// Returns logarithmic value of probalility (in the case of discrete distribution)
    /// or probability density (in the case of continuous distribution)
    /// </summary>
    public virtual double LogValue(double x)
    {
      return Math.Log(Value(x));
    }

    /// <summary>
    /// Fills distrubution parameters with Maximum Likelihood estimation
    /// </summary>
    public abstract void FromSample(double[] sample);

    /// <summary>
    /// Fills distrubution parameters with Maximum Likelihood estimation from a given classified sample.
    /// Ranges result with respect to classes and feature indices
    /// </summary>
    public abstract Dictionary<ClassFeatureKey, TParam> FromSample(ClassifiedSample<double[]> sample);
  }

  /// <summary>
  /// Base class for probability multidimensional distribution (discrete or continuous)
  /// </summary>
  public abstract class MultidimensionalDistributionBase<TParam> : IDistribution<TParam>
    where TParam : IDistributionParameters
  {
    protected MultidimensionalDistributionBase()
    {
    }

    /// <summary>
    /// Parameters of distribution
    /// </summary>
    public TParam Params { get; set; }

    /// <summary>
    /// Returns value of probalility (in the case of discrete distribution)
    /// or probability density (in the case of continuous distribution)
    /// </summary>
    public abstract double Value(double x);

    /// <summary>
    /// Returns logarithmic value of probalility (in the case of discrete distribution)
    /// or probability density (in the case of continuous distribution)
    /// </summary>
    public abstract double LogValue(double x);

    /// <summary>
    /// Fills distrubution parameters from sample (Maximum Likelihood estimation / frequency analysis etc)
    /// </summary>
    public abstract void FromSample(double[] sample);

    /// <summary>
    /// Fills distrubution parameters from given classified sample (Maximum Likelihood estimation / frequency analysis etc)
    /// Ranges result with respect to classes and feature indices
    /// </summary>
    public abstract Dictionary<ClassFeatureKey, TParam> FromSample(ClassifiedSample<double[]> sample);
  }
}
