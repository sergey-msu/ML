using System;
using System.Collections.Generic;
using ML.Core;

namespace ML.Contracts
{
  /// <summary>
  /// Contract for Informativity function
  /// </summary>
  public interface IInformativity
  {
    /// <summary>
    /// Caclulate maximum of informativity function
    /// </summary>
    Predicate<Point> Max(IEnumerable<Predicate<Point>> patterns, ClassifiedSample sample);
  }
}
