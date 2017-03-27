using System;
using System.Collections.Generic;
using ML.Contracts;

namespace ML.Core.Logical
{
  /// <summary>
  /// Base class for Informativity functions
  /// </summary>
  public abstract class IndexBase<TObj>: IInformativityIndex<TObj>, IMnemonicNamed
  {
    /// <summary>
    /// Index mnemonic ID
    /// </summary>
    public abstract string ID { get; }

    /// <summary>
    /// Index name
    /// </summary>
    public abstract string Name { get; }

    /// <summary>
    /// Calculates maximum of informativity function
    /// </summary>
    public Predicate<TObj> Max(IEnumerable<Predicate<TObj>> patterns, ClassifiedSample<TObj> sample)
    {
      Predicate<TObj> result = null;
      var maxIndex = double.MinValue;

      foreach (var pattern in patterns)
      {
        var index = Calculate(pattern, sample);
        if (index > maxIndex)
        {
          maxIndex = index;
          result = pattern;
        }
      }

      return result;
    }

    /// <summary>
    /// Calculates informativity function with respect to given pattern
    /// </summary>
    public abstract double Calculate(Predicate<TObj> pattern, ClassifiedSample<TObj> sample);
  }
}
