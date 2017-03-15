using System;
using System.Collections.Generic;
using ML.Core;

namespace ML.Contracts
{
  /// <summary>
  /// Contract for Informativity function
  /// </summary>
  public interface IInformIndex
  {
    /// <summary>
    /// Calculates maximum of informativity function
    /// </summary>
    Predicate<double[]> Max(IEnumerable<Predicate<double[]>> patterns, ClassifiedSample sample);

    /// <summary>
    /// Calculates informativity function with respect to given pattern
    /// </summary>
    double Calculate(Predicate<double[]> pattern, ClassifiedSample sample);
  }
}
