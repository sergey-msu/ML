using System;
using System.Collections.Generic;
using ML.Core;

namespace ML.Contracts
{
  /// <summary>
  /// Contract for Informativity function
  /// </summary>
  public interface IInformativityIndex<TObj>
  {
    /// <summary>
    /// Calculates maximum of informativity function
    /// </summary>
    Predicate<TObj> Max(IEnumerable<Predicate<TObj>> patterns, ClassifiedSample<TObj> sample);

    /// <summary>
    /// Calculates informativity function with respect to given pattern
    /// </summary>
    double Calculate(Predicate<TObj> pattern, ClassifiedSample<TObj> sample);
  }
}
