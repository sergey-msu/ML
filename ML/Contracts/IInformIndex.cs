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
    Predicate<Point> Max(IEnumerable<Predicate<Point>> patterns, ClassifiedSample sample);

    /// <summary>
    /// Calculates informativity function with respect to given pattern
    /// </summary>
    double Calculate(Predicate<Point> pattern, ClassifiedSample sample);
  }
}
