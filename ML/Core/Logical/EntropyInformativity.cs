using System;
using System.Collections.Generic;
using ML.Contracts;

namespace ML.Core.Logical
{
  public class EntropyInformativity : IInformativity
  {
    public struct EntropyBag
    {
      public float Pc;
      public float pc;
    }

    public Predicate<Point> Max(IEnumerable<Predicate<Point>> patterns, ClassifiedSample sample)
    {
      Predicate<Point> result = null;
      var maxEntropy = float.MinValue;

      //var info = new Dictionary<Class, EntropyBag>();

      //foreach (var pattern in patterns)
      //{
      //  var

      //  if (entropy > maxEntropy)
      //  {
      //    maxEntropy = entropy;
      //    result = pattern;
      //  }
      //}

      return result;
    }
  }
}
