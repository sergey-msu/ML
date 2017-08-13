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
  /// Contract for probability 1D distribution (discrete or continuous)
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
    /// Returns logarithmic value of probalility (in the case of discrete distribution)
    /// or probability density (in the case of continuous distribution)
    /// </summary>
    double LogValue(double x);

    /// <summary>
    /// Fills distrubution parameters from sample (Maximum Likelihood estimation / frequency analysis etc)
    /// </summary>
    void FromSample(double[] sample);

    /// <summary>
    /// Fills distrubution parameters from given classified sample (Maximum Likelihood estimation / frequency analysis etc)
    /// Ranges result with respect to classes and feature indices
    /// </summary>
    TParam[][] FromSample(ClassifiedSample<double[]> sample);
  }

  /// <summary>
  /// Contract for probability multidimensional distribution (discrete or continuous)
  /// </summary>
  public interface IMultidimensionalDistribution<TParam>
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
    double Value(double[] x);

    /// <summary>
    /// Returns logarithmic value of probalility (in the case of discrete distribution)
    /// or probability density (in the case of continuous distribution)
    /// </summary>
    double LogValue(double[] x);

    /// <summary>
    /// Fills distrubution parameters from sample (Maximum Likelihood estimation / frequency analysis etc)
    /// </summary>
    void FromSample(double[][] sample);

    /// <summary>
    /// Fills distrubution parameters from given classified sample (Maximum Likelihood estimation / frequency analysis etc)
    /// Ranges result with respect to classes and feature indices
    /// </summary>
    Dictionary<Class, TParam> FromSample(ClassifiedSample<double[]> sample);
  }
}
